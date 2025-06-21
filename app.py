import streamlit as st
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from Agents.Reason_agent import Reason_agent
from Task.Reason_task import Reason_task
from crewai import Crew
from utils.pdf_download import download_pdf_report
from utils.Astra_DB_search import get_astra_vectorstore, store_response

load_dotenv()

# --- Fetch API Key ---
api_key = os.getenv("STOCK_API")
if not api_key:
    st.error("❌ API Key not found. Please set it in your environment variables.")

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction with Chart")

# Initialize the vectorstore
vectorstore = get_astra_vectorstore()

# --- Input Section ---
st.markdown(
    """
    Enter a **valid stock symbol** (e.g., `AAPL`, `GOOG`, `TSLA`) below and click **Fetch & Predict**  
    to view the **last 10 days** of stock prices and predict the **next day’s close price** using an AI model.
    """
)

col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("🔎 Stock Symbol", placeholder="e.g., AAPL", help="Enter the stock symbol to analyze.")
with col2:
    st.markdown(" ")
    fetch_btn = st.button("🚀 Fetch & Predict", use_container_width=True)

if fetch_btn:
    with st.spinner("Fetching data..."):
        if not symbol:
            st.warning("⚠️ Please enter a stock symbol.")
        else:
            try:
                # --- Fetch Data ---
                url = "https://api.twelvedata.com/time_series"
                params = {
                    "symbol": symbol,
                    "interval": "1day",
                    "outputsize": 10,
                    "apikey": api_key
                }

                response = requests.get(url, params=params)
                data = response.json()

                if "values" not in data or not data["values"]:
                    st.error(f"❌ Error fetching data: {data.get('message', 'No data available for the given symbol.')}")
                    st.json(data)
                else:
                    df = pd.DataFrame(data["values"])[::-1]  # reverse chronological
                    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df["year"] = df["datetime"].dt.year
                    df["month"] = df["datetime"].dt.month
                    df["day"] = df["datetime"].dt.day
                    df["adjusted_close"] = df["close"]

                    # --- Preprocessing ---
                    features = ["open", "high", "low", "volume", "year", "month", "day"]
                    input_features = df[features].values

                    feature_scaler = MinMaxScaler()
                    target_scaler = MinMaxScaler()
                    feature_scaler.fit(input_features)
                    target_scaler.fit(df[["close"]].values)

                    model = load_model("model/best_model.keras", compile=False)
                    scaled_input = feature_scaler.transform(input_features).reshape(1, 10, 7)

                    # --- Prediction ---
                    predicted_scaled = model.predict(scaled_input)
                    predicted_actual = target_scaler.inverse_transform(predicted_scaled)[0][0]
                    last_close = df['close'].iloc[-1]
                    delta = predicted_actual - last_close
                    direction = "📈 Up" if delta > 0 else "📉 Down"

                    # Store the prediction in Astra DB
                    prediction_response = f"Predicted Close: ${predicted_actual:.2f}, Change: {direction}"
                    store_response(vectorstore, prediction_response, symbol)

                    # --- Display Results and Explanation ---
                    col1, col2 = st.columns(2)

                    # --- Prediction Section ---
                    with col1:
                        st.subheader("📊 Prediction Results")
                        st.metric(label="Predicted Close (Next Day)", value=f"${predicted_actual:.2f}", delta=f"{delta:.2f}", delta_color="normal")
                        st.metric(label="Last Actual Close", value=f"${last_close:.2f}")
                        st.markdown(f"**📊 Change:** `{direction}`")

                        # --- Plot ---
                        future_date = df["datetime"].iloc[-1] + timedelta(days=1)
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(df["datetime"], df["close"], marker='o', color="#1f77b4", label="Actual Close Price", linewidth=2)
                        ax.plot([df["datetime"].iloc[-1], future_date], [df["close"].iloc[-1], predicted_actual],
                                linestyle='--', color="#ff7f0e", linewidth=2, alpha=0.7)
                        ax.scatter(future_date, predicted_actual, color="#d62728", s=120, marker='*', label="Predicted Next Day", zorder=5)
                        ax.annotate(f"{predicted_actual:.2f}", (future_date, predicted_actual), textcoords="offset points",
                                    xytext=(0, 10), ha='center', fontsize=12, color="#d62728", fontweight='bold')

                        # ±3% band
                        lower = predicted_actual * 0.97
                        upper = predicted_actual * 1.03
                        ax.fill_between([df["datetime"].iloc[-1], future_date], [lower, lower], [upper, upper],
                                        color="orange", alpha=0.1, label="±3% Confidence Band")
                        ax.axvline(future_date, linestyle=':', color='gray', alpha=0.5)
                        ax.set_title(f"{symbol.upper()} Stock - Last 10 Days + Prediction", fontsize=16, fontweight='bold')
                        ax.set_xlabel("Date", fontsize=12)
                        ax.set_ylabel("Price", fontsize=12)
                        ax.legend(fontsize=12)
                        ax.grid(True, linestyle='--', alpha=0.6)
                        fig.autofmt_xdate()
                        st.pyplot(fig)

                    # --- Explanation Section ---
                    with col2:
                        st.subheader("🤖 Explanation")
                        with st.spinner("🧠 Reasoning about the prediction..."):
                            # Convert predicted_actual to a standard Python float
                            crew = Crew(agents=[Reason_agent], tasks=[Reason_task])
                            result = crew.kickoff({"predicted_actual": float(predicted_actual), "symbol": symbol})

                        st.markdown("### Explanation of Prediction")
                        st.info(result, icon="💡")
                        download_pdf_report(result, filename_prefix=f"{symbol}_prediction_report", is_markdown=True)

            except Exception as e:
                st.error(f"❗ Exception occurred: `{e}`")
