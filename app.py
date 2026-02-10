import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Load model & scaler
model = load_model("model (1).h5")

with open("scaler(2).pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("✈️ Airline Passenger Forecasting (LSTM)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df.head())

    df.columns = df.columns.str.strip()
    data = df[['Passengers']].values.astype("float32")
    scaled = scaler.transform(data)

    look_back = 1
    X = []

    for i in range(len(scaled)-look_back):
        X.append(scaled[i:i+look_back, 0])

    X = np.array(X)
    X = X.reshape(X.shape[0], look_back, 1)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    st.subheader("Prediction Plot")
    plt.figure(figsize=(10,5))
    plt.plot(df["Passengers"].values, label="Actual")
    plt.plot(range(look_back, len(predictions)+look_back), predictions, label="Predicted")
    plt.legend()
    st.pyplot(plt)


    # Convert predictions into 1D array
    pred = predictions.reshape(-1)

    # Actual values (same length as predictions)
    actual = df["Passengers"].values[look_back:look_back + len(pred)]

    # Convert actual into float
    actual = actual.astype("float32")

    # Remove NaN values if present
    mask = ~np.isnan(actual) & ~np.isnan(pred)

    actual_clean = actual[mask]
    pred_clean = pred[mask]

    st.subheader("📌 Model Evaluation")

    # Check if data is valid
    if len(actual_clean) == 0 or len(pred_clean) == 0:
        st.error("❌ Cannot calculate RMSE/MAE because data contains NaN values.")
    else:
        rmse = math.sqrt(mean_squared_error(actual_clean, pred_clean))
        mae = mean_absolute_error(actual_clean, pred_clean)

        st.write(f"✅ RMSE: {rmse:.2f}")
        st.write(f"✅ MAE : {mae:.2f}")

        st.write(f"📌 Total Points: {len(actual_clean)}")
        st.write(f"📌 NaN Removed: {len(actual) - len(actual_clean)}")

st.subheader("📌 About This Project")

st.write("""
This project is a **Time Series Forecasting Web Application** built using **LSTM (Long Short-Term Memory)**.
It predicts airline passenger demand based on past historical passenger data.

### 🔥 Key Features:
- Upload your own CSV dataset
- Preprocessing using MinMaxScaler
- LSTM-based forecasting model
- Graph visualization (Actual vs Predicted)
- Model evaluation using RMSE and MAE

### 🛠️ Tech Stack:
- Python
- TensorFlow / Keras
- Streamlit
- Pandas, NumPy
- Matplotlib
- Scikit-learn

### 🎯 Use Case:
This type of forecasting is useful for:
- Airline demand prediction
- Sales forecasting
- Inventory planning
- Business decision making
""")



