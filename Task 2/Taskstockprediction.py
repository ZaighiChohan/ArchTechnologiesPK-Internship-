# Step 1: Loading the Apple stock dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
# Load raw CSV
df = pd.read_csv(r"C:\Users\ZAIGHI CHOHAN\Downloads\ArchTechnologiesPK\apple_share_price.csv")

# Preview
print("Shape:", df.shape)
df.head()

# Parse 'Date' column into datetime
df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%y")

# Sort ascending by date
df = df.sort_values("Date").reset_index(drop=True)

# Check again
print (df.head())


# visualization
plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['Close'])
plt.title("Apple Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Plotting volume
plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['Volume'], color="orange")
plt.title("Apple Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()



df['Return'] = df['Close'].pct_change()# Step 2: Add daily returns

# Plot returns
plt.figure(figsize=(10,4))
df['Return'].hist(bins=100, alpha=0.7)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df[['Open','High','Low','Close','Volume','Return']].corr(),
            annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# Rolling mean and volatility
df['RollingMean'] = df['Close'].rolling(window=20).mean()
df['Volatility'] = df['Return'].rolling(window=20).std()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label="Close")
plt.plot(df['Date'], df['RollingMean'], label="20-Day MA")
plt.legend()
plt.title("Apple Closing Price & 20-Day Moving Average")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Volatility'], color="red")
plt.title("20-Day Rolling Volatility")
plt.show()


# Step 2: EDA – Returns, Correlations, Rolling Stats

df['Return'] = df['Close'].pct_change()

# Distribution of returns
plt.figure(figsize=(10,4))
df['Return'].hist(bins=100, alpha=0.7)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return"); plt.ylabel("Frequency")
plt.show()

# Correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(df[['Open','High','Low','Close','Volume','Return']].corr(),
            annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# Rolling averages and volatility
df['RollingMean'] = df['Close'].rolling(window=20).mean()
df['Volatility'] = df['Return'].rolling(window=20).std()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label="Close")
plt.plot(df['Date'], df['RollingMean'], label="20-Day MA")
plt.legend(); plt.title("Apple Close & 20-Day Moving Average")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Volatility'], color="red")
plt.title("20-Day Rolling Volatility")
plt.show()

#---------------------------------------------
# ===============================================
# =========================
# LSTM (robust version)
# =========================

# 1) Prepare arrays
features = ['Open','High','Low','Close','Volume']
Xraw = df[features].values.astype('float32')
yraw = df[['Close']].values.astype('float32')   # shape (N,1)
dates = df['Date'].values

# chronological train/test split (80/20)
split = int(len(Xraw)*0.8)

# --- fit scalers on TRAIN ONLY (avoid leakage)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_raw, X_test_raw = Xraw[:split], Xraw[split:]
y_train_raw, y_test_raw = yraw[:split], yraw[split:]

X_train = scaler_X.fit_transform(X_train_raw)
X_test  = scaler_X.transform(X_test_raw)
y_train = scaler_y.fit_transform(y_train_raw)
y_test  = scaler_y.transform(y_test_raw)

# 2) Build sequences (past seq_len days -> predict next day's Close)
def make_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])  # predict t from [t-seq_len, t-1]
    return np.array(Xs, dtype='float32'), np.array(ys, dtype='float32')

seq_len = 30

# sequences for full set, then split so dates align nicely
X_all = np.vstack([X_train, X_test])
y_all = np.vstack([y_train, y_test])
X_seq, y_seq = make_sequences(X_all, y_all, seq_len=seq_len)

# index in the *sequenced* arrays where test starts
split_seq = split - seq_len
Xtr, Xte = X_seq[:split_seq], X_seq[split_seq:]
ytr, yte = y_seq[:split_seq], y_seq[split_seq:]

dates_seq = dates[seq_len:]
dates_te = dates_seq[split_seq:]

print("Shapes ->", Xtr.shape, ytr.shape, Xte.shape, yte.shape)

# 3) PyTorch tensors & loaders
device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
test_ds  = TensorDataset(torch.tensor(Xte), torch.tensor(yte))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)  # keep temporal order
test_X = torch.tensor(Xte).to(device)
test_y = torch.tensor(yte).to(device)

# 4) Model
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)

model = LSTMRegressor(n_features=Xtr.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5) Train
epochs = 80
for epoch in range(epochs):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()*len(xb)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} - train_loss={running/len(train_ds):.6f}")

# 6) Predict (scaled) → inverse transform back to price scale
model.eval()
with torch.no_grad():
    y_pred_scaled = model(test_X).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)[:,0]
y_true = scaler_y.inverse_transform(test_y.cpu().numpy())[:,0]

# 7) Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print("\nLSTM (scaled targets) results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 8) Plot
plt.figure(figsize=(12,6))
plt.plot(dates_te, y_true, label="Actual")
plt.plot(dates_te, y_pred, label="Predicted")
plt.title("LSTM — Actual vs Predicted Close (inverse-transformed)")
plt.xlabel("Date"); plt.ylabel("Close Price")
plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(dates_te[-100:], y_true[-100:], label="Actual")
plt.plot(dates_te[-100:], y_pred[-100:], label="Predicted")
plt.title("LSTM — Zoomed View (Last 100 Days)")
plt.xlabel("Date"); plt.ylabel("Close Price")
plt.legend(); plt.tight_layout()
plt.show()

errors = y_true - y_pred
plt.figure(figsize=(8,5))
plt.hist(errors, bins=40, color="purple", alpha=0.7)
plt.axvline(0, color="black", linestyle="--")
plt.title("Distribution of Prediction Errors (Residuals)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(dates_te, errors, color="red")
plt.axhline(0, linestyle="--", color="black")
plt.title("Residuals Over Time")
plt.xlabel("Date"); plt.ylabel("Error")
plt.show()
