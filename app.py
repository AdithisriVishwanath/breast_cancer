# app.py

import streamlit as st
import numpy as np
import joblib
from sklearn import datasets
import os

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Load model and scaler
st.subheader("🧠 Loading model and scaler...")

model_path = "mlp_2layer_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Required model or scaler file is missing!")
    st.stop()

mlp = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load dataset to get feature names
data = datasets.load_breast_cancer()
X_train = data.data
feature_names = data.feature_names

st.title("🩺 Breast Cancer Predictor (2-Layer MLP)")
st.write("Enter the following values to get a prediction.")

# Input fields
inputs = []
for i, feature in enumerate(feature_names):
    col_data = X_train[:, i]
    val = st.number_input(
        f"{feature}",
        min_value=float(np.min(col_data)),
        max_value=float(np.max(col_data)),
        value=float(np.mean(col_data)),
        step=0.01,
        format="%.4f"
    )
    inputs.append(val)

if st.button("Predict"):
    try:
        inputs_scaled = scaler.transform([inputs])
        prediction = mlp.predict(inputs_scaled)
        result = 'Malignant' if prediction[0] == 0 else 'Benign'
        st.success(f"🎯 Prediction: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
