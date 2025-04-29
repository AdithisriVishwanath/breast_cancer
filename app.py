import streamlit as st
import numpy as np
import joblib
from sklearn import datasets

# Load model and scaler
scaler = joblib.load('scaler.pkl')
mlp = joblib.load('mlp_2layer_model.pkl')

# Load feature names
data = datasets.load_breast_cancer()

st.title("Breast Cancer Predictor (2-Layer MLP)")

# Input fields for each feature
inputs = []
for i, feature in enumerate(data.feature_names):
    val = st.number_input(f"{feature}", value=float(data.data[:, i].mean()))
    inputs.append(val)

# Predict button
if st.button("Predict"):
    inputs_scaled = scaler.transform([inputs])  # Scale the inputs
    prediction = mlp.predict(inputs_scaled)     # Predict

    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    st.success(f"Prediction: {result}")
