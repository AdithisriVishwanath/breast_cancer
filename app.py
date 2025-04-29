#writefile app.py
import streamlit as st
import joblib
import numpy as np
from sklearn import datasets  # You missed this in your code

# Load the dataset (this was missing)
data = datasets.load_breast_cancer()
X_train = data.data

st.title("Breast Cancer Predictor (2-Layer MLP)")

# Input fields for features
inputs = []
for i, feature in enumerate(data.feature_names):
    val = st.number_input(f"{feature}", value=float(X_train[:, i].mean()))
    inputs.append(val)

if st.button("Predict"):
    inputs_scaled = scaler.transform([inputs])
    prediction = mlp.predict(inputs_scaled)
    st.success(f"Prediction: {'Malignant' if prediction[0] == 0 else 'Benign'}")
