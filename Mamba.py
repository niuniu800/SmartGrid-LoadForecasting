# ============================
#        Imports
# ============================
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================
#        KANLinear Layer
# ============================
class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] +
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        return (base_output + spline_output).view(*original_shape[:-1], self.out_features)


# ============================
#        Mamba Network
# ============================
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    features: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight



class MambaBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_proj = KANLinear(args.d_model, args.d_inner * 2)
        self.conv1d = nn.Conv1d(args.d_inner, args.d_inner, kernel_size=args.d_conv, groups=args.d_inner,
                                bias=args.conv_bias, padding=args.d_conv - 1)
        self.x_proj = KANLinear(args.d_inner, args.dt_rank + args.d_state * 2)
        self.dt_proj = KANLinear(args.dt_rank, args.d_inner)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        b, l, _ = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(self.args.d_inner, dim=-1)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        y = self.ssm(x) * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log)
        D = self.D
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')
        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        return torch.stack(ys, dim=1) + u * D


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class Mamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encode = nn.Linear(args.features, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.decode = nn.Linear(args.d_model, 1)

    def forward(self, input_ids):
        x = self.encode(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.decode(x).squeeze(-1)
# ============================
#        Data Handling
# ============================
look_back = 1
T = 1
batch_size = 32
learn_rate = 0.001
epochs = 15

# Load and normalize data
dataset = pd.read_csv(r'D:\Desktop\2025\先导杯\quanzhou.csv', usecols=[1, 2, 3, 4, 5])
dataX = dataset.values
dataY = dataset['FUHE'].values  # 注意：'FUHE'列似乎未包含在上面 usecols 中

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
data_X = scaler1.fit_transform(dataX)
data_Y = scaler2.fit_transform(dataY.reshape(-1, 1))

train_size = int(len(data_X) * 0.7)
val_size = int(len(data_X) * 0.1)
test_size = len(data_X) - train_size - val_size
train_X, train_Y = data_X[:train_size], data_Y[:train_size]
val_X, val_Y = data_X[train_size:train_size+val_size], data_Y[train_size:train_size+val_size]
test_X, test_Y = data_X[train_size+val_size:], data_Y[train_size+val_size:]

def create_dataset(datasetX, datasetY, look_back, T):
    dataX, dataY = [], []
    for i in range(0, len(datasetX) - look_back - T, T):
        a = datasetX[i:(i + look_back), :]
        dataX.append(a)
        if T == 1:
            dataY.append(datasetY[i + look_back])
        else:
            dataY.append(datasetY[i + look_back:i + look_back + T, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train_X, train_Y, look_back, T)
valX, valY = create_dataset(val_X, val_Y, look_back, T)
testX, testY = create_dataset(test_X, test_Y, look_back, T)

trainX = torch.Tensor(trainX)
trainY = torch.Tensor(trainY)
valX = torch.Tensor(valX)
valY = torch.Tensor(valY)
testX = torch.Tensor(testX)
testY = torch.Tensor(testY)

class MyDataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y
    def __getitem__(self, index):
        return self.data_X[index], self.data_Y[index]
    def __len__(self):
        return len(self.data_X)

train_loader = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(MyDataset(valX, valY), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(MyDataset(testX, testY), batch_size=batch_size, shuffle=False)

# ============================
#         Training
# ============================
args = ModelArgs(d_model=128, n_layer=4, features=trainX.shape[2])
model = Mamba(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    # Early Stopping 设置
    patience = 5  # 容忍连续多少个 epoch 无提升
    best_val_loss = float('inf')
    patience_counter = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            val_loss = criterion(model(inputs), labels)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
    #     torch.save(model.state_dict(), 'best_mamba_model.pth')
    #     print("Model saved")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_mamba_model.pth')
        print(f"Model saved at epoch {epoch + 1}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('mamba_loss_curve.png')
plt.show()

# ============================
#           Test
# ============================
model.load_state_dict(torch.load('best_mamba_model.pth'))
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)

predictions = scaler2.inverse_transform(predictions)
actuals = scaler2.inverse_transform(actuals)

r2 = r2_score(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print(f'R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')

plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Power Load/KW')
plt.title('Power Load Prediction')
plt.legend()
plt.savefig('prediction_vs_actual.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5)
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Scatter Plot: Actual vs Predicted')
plt.savefig('scatter_actual_predicted.png')
plt.show()
