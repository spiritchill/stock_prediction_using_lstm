import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = load_model("lstm_stock_model.keras")
    
    # Reconstruct scaler from numpy arrays — no pickle, no version issues
    scaler = MinMaxScaler()
    scaler.data_min_  = np.load("scaler_min.npy")
    scaler.data_max_  = np.load("scaler_max.npy")
    scaler.scale_     = 1 / (scaler.data_max_ - scaler.data_min_)
    scaler.data_range_= scaler.data_max_ - scaler.data_min_
    scaler.min_       = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = len(scaler.data_min_)
    scaler.n_samples_seen_ = 1
    scaler.feature_names_in_ = None
    return model, scaler

model, scaler = load_assets()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")
st.title("LSTM Stock Price Predictor")
st.caption("Predict next-day closing prices using a stacked LSTM neural network.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker        = st.text_input("Stock Ticker", value="AAPL").upper()
    period        = st.selectbox("Historical Period", ["2y", "3y", "5y", "10y"], index=2)
    forecast_days = st.slider("Forecast Days", 1, 30, 7)
    run_btn       = st.button("Predict", use_container_width=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = ["Close", "Volume", "Return", "MA_10", "MA_50", "Volatility"]
LOOKBACK  = 60   # must match what was used in save_model.py

# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_and_engineer(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True)
    # Fix MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df["Return"]     = df["Close"].pct_change()
    df["MA_10"]      = df["Close"].rolling(10).mean()
    df["MA_50"]      = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df.dropna(inplace=True)
    return df

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def inverse_close(scaler, scaled_vals):
    dummy = np.zeros((len(scaled_vals), len(FEATURES)))
    dummy[:, 0] = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]

# ── Main ──────────────────────────────────────────────────────────────────────
if run_btn:
    # 1. Fetch data
    with st.spinner(f"Fetching {ticker} data..."):
        try:
            df = fetch_and_engineer(ticker, period)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    st.success(f"✅ Model loaded! Predicting for **{ticker}**...")

    # 2. Scale using the LOADED scaler (not a new one)
    scaled = scaler.transform(df[FEATURES].values)
    split  = int(len(scaled) * 0.8)

    X_test, y_test = create_sequences(scaled[split:], LOOKBACK)

    # 3. Predict
    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred_price  = inverse_close(scaler, pred_scaled)
    real_price  = inverse_close(scaler, y_test)

    # 4. Metrics
    rmse = np.sqrt(mean_squared_error(real_price, pred_price))
    mae  = mean_absolute_error(real_price, pred_price)
    mape = np.mean(np.abs((real_price - pred_price) / real_price)) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"${rmse:.2f}")
    col2.metric("MAE",  f"${mae:.2f}")
    col3.metric("MAPE", f"{mape:.2f}%")

    # 5. Actual vs Predicted chart
    test_dates = df.index[split + LOOKBACK:]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=test_dates, y=real_price, name="Actual",
                              line=dict(color="#1f77b4", width=2)))
    fig1.add_trace(go.Scatter(x=test_dates, y=pred_price, name="Predicted",
                              line=dict(color="#ff7f0e", width=2, dash="dot")))
    fig1.update_layout(title=f"{ticker} — Actual vs Predicted",
                       xaxis_title="Date", yaxis_title="Price (USD)",
                       hovermode="x unified", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # 6. Forecast next N days
    st.subheader(f"🔮 {forecast_days}-Day Forecast")
    window = scaled[-LOOKBACK:].copy()
    future_prices = []

    for _ in range(forecast_days):
        x = window[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURES))
        p = model.predict(x, verbose=0)[0, 0]
        future_prices.append(p)
        new_row    = window[-1].copy()
        new_row[0] = p
        window     = np.vstack([window, new_row])

    future_prices = inverse_close(scaler, np.array(future_prices))
    future_dates  = pd.bdate_range(start=df.index[-1], periods=forecast_days + 1)[1:]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index[-60:], y=df["Close"].values[-60:],
                              name="Recent Actual", line=dict(color="#1f77b4", width=2)))
    fig2.add_trace(go.Scatter(x=future_dates, y=future_prices,
                              name="Forecast", line=dict(color="#9467bd", width=2, dash="dash"),
                              mode="lines+markers"))
    fig2.update_layout(title=f"{ticker} — {forecast_days}-Day Forecast",
                       xaxis_title="Date", yaxis_title="Price (USD)",
                       hovermode="x unified", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # 7. Forecast table
    forecast_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Forecast Price (USD)": future_prices.round(2)
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Pick a ticker and click **Predict** to start.")
