
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Download Data ───────────────────────────────────────────────────────
def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.dropna(inplace=True)
    return df

# ─── 2. Feature Engineering ─────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"]    = df["Close"].pct_change()
    df["MA_10"]     = df["Close"].rolling(10).mean()
    df["MA_50"]     = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df.dropna(inplace=True)
    return df

# ─── 3. Build Sequences ──────────────────────────────────────────────────────
def create_sequences(data: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])          # predict scaled Close price
    return np.array(X), np.array(y)

# ─── 4. Model ────────────────────────────────────────────────────────────────
def build_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")   # Huber loss = robust to outliers
    return model

# ─── 5. Train / Evaluate ─────────────────────────────────────────────────────
def train_and_evaluate(ticker: str = "AAPL"):
    # --- data
    df      = add_features(fetch_stock_data(ticker))
    features = ["Close", "Volume", "Return", "MA_10", "MA_50", "Volatility"]
    raw     = df[features].values

    # --- scale
    scaler  = MinMaxScaler()
    scaled  = scaler.fit_transform(raw)

    # --- split (80 / 20, no shuffle – time series!)
    split   = int(len(scaled) * 0.8)
    train   = scaled[:split]
    test    = scaled[split:]

    LOOKBACK = 60
    X_train, y_train = create_sequences(train, LOOKBACK)
    X_test,  y_test  = create_sequences(test,  LOOKBACK)

    # --- model
    model = build_model((LOOKBACK, len(features)))
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # --- predict & inverse-transform Close price only
    pred_scaled = model.predict(X_test)

    # Reconstruct full rows to inverse-transform
    dummy = np.zeros((len(pred_scaled), len(features)))
    dummy[:, 0] = pred_scaled.flatten()
    pred_price = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((len(y_test), len(features)))
    dummy2[:, 0] = y_test
    real_price = scaler.inverse_transform(dummy2)[:, 0]

    # --- metrics
    rmse = np.sqrt(mean_squared_error(real_price, pred_price))
    mae  = mean_absolute_error(real_price, pred_price)
    mape = np.mean(np.abs((real_price - pred_price) / real_price)) * 100
    print(f"\nRMSE : ${rmse:.2f}")
    print(f"MAE  : ${mae:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # --- plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(real_price, label="Actual",    color="#1f77b4", linewidth=1.5)
    axes[0].plot(pred_price, label="Predicted", color="#ff7f0e", linewidth=1.5, alpha=0.8)
    axes[0].set_title(f"{ticker} – Actual vs Predicted Close Price")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Training History")
    axes[1].set_ylabel("Huber Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model, scaler, pred_price, real_price

# ─── 6. Forecast next N days ─────────────────────────────────────────────────
def forecast_next_days(model, scaler, df, features, n_days=5, lookback=60):
    raw    = df[features].values
    scaled = scaler.transform(raw)
    window = scaled[-lookback:]       # last `lookback` rows

    preds = []
    for _ in range(n_days):
        x   = window[-lookback:].reshape(1, lookback, len(features))
        p   = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        new_row      = window[-1].copy()
        new_row[0]   = p            # update Close
        window       = np.vstack([window, new_row])

    dummy = np.zeros((n_days, len(features)))
    dummy[:, 0] = preds
    prices = scaler.inverse_transform(dummy)[:, 0]

    print(f"\nNext {n_days}-day forecast:")
    for i, p in enumerate(prices, 1):
        print(f"  Day +{i}: ${p:.2f}")
    return prices

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TICKER   = "AAPL"          # change to any ticker
    features = ["Close", "Volume", "Return", "MA_10", "MA_50", "Volatility"]

    model, scaler, pred, real = train_and_evaluate(TICKER)

    df = add_features(fetch_stock_data(TICKER))
    forecast_next_days(model, scaler, df, features, n_days=5)
