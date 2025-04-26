import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the model (make sure to give the correct path to your saved model)
model = tf.keras.models.load_model('C:\Users\adith\OneDrive\Documents\sems 6\mlops\mlopMP\breast_cancer_model.h5')

# Define the Streamlit app
st.title('Breast Cancer Prediction')

# Input fields for the user to enter features
radius = st.number_input("Radius", min_value=0.0, max_value=100.0, value=0.0)
texture = st.number_input("Texture", min_value=0.0, max_value=100.0, value=0.0)
perimeter = st.number_input("Perimeter", min_value=0.0, max_value=100.0, value=0.0)
area = st.number_input("Area", min_value=0.0, max_value=1000.0, value=0.0)
smoothness = st.number_input("Smoothness", min_value=0.0, max_value=1.0, value=0.0)

# Convert the inputs into a format the model can work with
features = np.array([[radius, texture, perimeter, area, smoothness]])

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Predict using the model
if st.button('Predict'):
    prediction = model.predict(features)
    prediction_label = 'Malignant' if prediction[0] > 0.5 else 'Benign'
    st.write(f"The model predicts the tumor is: {prediction_label}")

