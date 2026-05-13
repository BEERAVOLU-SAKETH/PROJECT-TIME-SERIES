import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARMA Anomaly Detection Dashboard", layout="wide")

st.title("📊 Real-Time Anomaly Detection Dashboard")
st.write("ARMA-based anomaly detection and early warning system using residual analysis.")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()

    st.subheader("Raw Data")
    st.line_chart(df["value"])

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.header("Model Settings")

    p = st.sidebar.selectbox("AR order (p)", [1, 2, 3], index=1)
    q = st.sidebar.selectbox("MA order (q)", [1, 2, 3], index=1)

    anomaly_k = st.sidebar.slider(
        "Anomaly Threshold (k × σ)",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.1
    )

    warning_k = st.sidebar.slider(
        "Warning Threshold (k × σ)",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

    consecutive_points = st.sidebar.slider(
        "Consecutive Points for Warning",
        min_value=2,
        max_value=8,
        value=3,
        step=1
    )

    # -----------------------------
    # Train-test split
    # -----------------------------
    train_size = int(len(df) * 0.7)

    train = df["value"].iloc[:train_size]
    test = df["value"].iloc[train_size:]

    # -----------------------------
    # Fit ARMA model
    # ARMA(p, q) = ARIMA(p, 0, q)
    # -----------------------------
    with st.spinner("Training ARMA model..."):
        model = ARIMA(df["value"], order=(p, 0, q))
        model_fit = model.fit()

    # -----------------------------
    # One-step-ahead prediction
    # This avoids flat long-horizon forecasts
    # -----------------------------
    predictions = model_fit.predict(
        start=test.index[0],
        end=test.index[-1],
        dynamic=False
    )

    predictions = pd.Series(predictions, index=test.index)

    # -----------------------------
    # Residual calculation
    # Z_t = X_t - X_hat_t
    # -----------------------------
    residuals = test - predictions

    # Use training residuals to calculate sigma
    train_predictions = model_fit.predict(
        start=train.index[0],
        end=train.index[-1],
        dynamic=False
    )

    train_residuals = train - train_predictions
    sigma = np.std(train_residuals)

    anomaly_threshold = anomaly_k * sigma
    warning_threshold = warning_k * sigma

    # -----------------------------
    # Detect anomalies
    # -----------------------------
    anomalies = np.where(np.abs(residuals) > anomaly_threshold)[0]

    # -----------------------------
    # Detect early warnings
    # -----------------------------
    warnings = []

    residual_array = residuals.values

    for i in range(consecutive_points - 1, len(residual_array)):
        recent_window = residual_array[i - consecutive_points + 1 : i + 1]

        if np.all(np.abs(recent_window) > warning_threshold):
            warnings.append(i)

    # -----------------------------
    # Metrics Cards
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training Points", len(train))
    col2.metric("Testing Points", len(test))
    col3.metric("Detected Anomalies", len(anomalies))
    col4.metric("Early Warnings", len(warnings))

    st.write("Estimated residual standard deviation σ:", round(float(sigma), 4))
    st.write("Anomaly threshold:", round(float(anomaly_threshold), 4))
    st.write("Warning threshold:", round(float(warning_threshold), 4))

    # -----------------------------
    # Main Detection Plot
    # -----------------------------
    st.subheader("Actual vs Predicted with Anomalies and Early Warnings")

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(test.index, test.values, label="Actual Data", linewidth=1.5)
    ax.plot(test.index, predictions.values, label="Predicted Data", linewidth=1.5)

    ax.scatter(
        test.index[anomalies],
        test.iloc[anomalies],
        color="red",
        label="Anomalies",
        s=60,
        zorder=5
    )

    ax.scatter(
        test.index[warnings],
        test.iloc[warnings],
        color="yellow",
        label="Early Warnings",
        s=45,
        zorder=4
    )

    ax.set_title("Real-Time Anomaly Detection & Early Warning System")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # -----------------------------
    # Residual Plot
    # -----------------------------
    st.subheader("Residual Analysis")

    fig2, ax2 = plt.subplots(figsize=(16, 5))

    ax2.plot(test.index, residuals.values, label="Residuals")
    ax2.axhline(anomaly_threshold, linestyle="--", label="+ Anomaly Threshold")
    ax2.axhline(-anomaly_threshold, linestyle="--", label="- Anomaly Threshold")
    ax2.axhline(warning_threshold, linestyle=":", label="+ Warning Threshold")
    ax2.axhline(-warning_threshold, linestyle=":", label="- Warning Threshold")

    ax2.set_title("Residuals with Thresholds")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Residual")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

    # -----------------------------
    # Result Table
    # -----------------------------
    st.subheader("Detected Anomaly Timestamps")

    if len(anomalies) > 0:
        anomaly_table = pd.DataFrame({
            "Timestamp": test.index[anomalies],
            "Actual Value": test.iloc[anomalies].values,
            "Predicted Value": predictions.iloc[anomalies].values,
            "Residual": residuals.iloc[anomalies].values
        })

        st.dataframe(anomaly_table)
    else:
        st.success("No anomalies detected at the selected threshold.")

else:
    st.info("Upload the NAB machine_temperature_system_failure.csv file to begin.")