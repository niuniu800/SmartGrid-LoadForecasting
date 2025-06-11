#  SmartGrid-LoadForecasting

Power load forecasting using deep learning models: a comparative study of Transformer and Mamba architectures for smart grid applications.

---

##  Overview

This project focuses on short-term electric load forecasting for smart grids using two state-of-the-art deep learning architectures:

- **Transformer**: A self-attention-based model capable of capturing short-term dependencies in time series.
- **Mamba + KAN**: A novel state space model enhanced with Kolmogorov–Arnold Networks (KAN) for modeling complex, nonlinear, and long-range temporal dependencies.

Both models are implemented in PyTorch and trained on real-world power grid data (`quanzhou.csv`), aiming to improve load prediction accuracy under nonstationary and highly dynamic conditions.

---

##  Model Architectures

### ⚙️ Transformer
- Encoder-decoder structure (3 layers each)
- Multi-head attention (4 heads)
- Layer normalization, residual connections
- Fully connected output for single-step regression

###  Mamba + KAN
- State Space Modeling (SSM) with Selective Scan mechanism
- KANLinear layers for flexible nonlinear activations
- Captures long-term dependencies and sharp load transitions
- Includes early stopping to avoid overfitting

---

##  Experimental Results

Both models were trained on the same dataset and evaluated using standard metrics:

| Model       | R²     | MAE      | RMSE     | MAPE    |
|-------------|--------|----------|----------|---------|
| Transformer | 0.8529 | 590.52   | 759.94   | 6.64%   |
| Mamba       | 0.9814 | 169.39   | 269.92   | 2.05%   |

Mamba showed superior performance in all metrics, especially under rapid load fluctuations. It also better fits actual trends and generalizes more robustly.

---

##  File Structure

```bash
SmartGrid-LoadForecasting/
├── Transformer.py             # Transformer model script
├── Mamba.py                   # Mamba + KAN model script
├── quanzhou.csv               # Input dataset (not uploaded here)
├── requirements.txt           # Python package dependencies
├── prediction_vs_actual.png   # Output plot (Mamba)
├── trans_1.png / trans_2.png  # Output plots (Transformer)
└── README.md                  # Project description
Getting Started
1. Install Dependencies

pip install torch numpy pandas matplotlib scikit-learn einops tqdm
2. Run the Models
Transformer:
python Transformer.py

Mamba:
python Mamba.py

 Make sure the file path to quanzhou.csv is correctly set in the scripts. By default, it points to a Windows local path (you may need to change it).

Dataset
Source: Provided by the competition (or organization)

Features: 5 input variables + target variable (FUHE)

Preprocessing: Normalized using MinMaxScaler

Sample generation: sliding window (look_back = 1, T = 1)

Visualization
prediction_vs_actual.png: Predicted vs actual load (Mamba)

trans_1.png: Transformer prediction curve

trans_2.png: Scatter plot of actual vs predicted (Transformer)

License
This project is for educational and academic research purposes only.
