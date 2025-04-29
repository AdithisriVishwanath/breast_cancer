# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn import datasets

# Load model and scaler


# Load the dataset
data = datasets.load_breast_cancer()
X_train = data.data
feature_names = data.feature_names

st.title("🩺 Breast Cancer Predictor (2-Layer MLP)")
st.write("Enter the following parameters to predict whether the tumor is benign or malignant.")

# Input fields
inputs = []
for i, feature in enumerate(feature_names):
    feature_data = X_train[:, i]
    val = st.number_input(
        f"{feature}",
        min_value=float(np.min(feature_data)),
        max_value=float(np.max(feature_data)),
        value=float(np.mean(feature_data)),
        step=0.01,
        format="%.4f"
    )
    inputs.append(val)

# Predict button
if st.button("Predict"):
    inputs_scaled = scaler.transform([inputs])
    prediction = mlp.predict(inputs_scaled)
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    st.success(f"🎯 Prediction: **{result}**")
