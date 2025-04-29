# breast_cancer

"https://qnq9i8mfkcpgin95dkvfxh.streamlit.app/" url for the app

###Predicting malignant versus benign breast cancer cases using MLP and Streamlit  
Predicting whether a diagnosed breast cancer case is **malignant or benign** using the **Wisconsin Breast Cancer Dataset**. The model is trained with a **Multilayer Perceptron (MLP)** neural network and deployed using a **Streamlit** web app for real-time predictions.

### Project Overview  
According to the World Health Organization and CDC, breast cancer is among the most prevalent forms of cancer among women globally. Early and accurate detection is crucial for timely intervention. This project uses **diagnostic features of breast masses** derived from digitized images to train an MLP-based classifier. A frontend interface via Streamlit allows users to enter tumor characteristics and receive a prediction.

#### Files in this repository  
- `train_and_save_model.py` — trains the MLP model and saves it along with the scaler as `.pkl` files  
- `app.py` — Streamlit application for real-time predictions  
- `mlp_2layer_model.pkl` — trained MLP model  
- `scaler.pkl` — StandardScaler used to normalize input features  
- `README.md` — detailed explanation of the project  

#### Problem Statement  
This project investigates the ability of an MLP classifier to predict the **malignancy** of a breast tumor using 30 numerical features extracted from digitized medical images.  
Steps included in the approach:  
•	Load the Breast Cancer Wisconsin Diagnostic dataset using sklearn  
•	Explore the features and visualize distributions  
•	Standardize feature inputs using StandardScaler  
•	Train a Multilayer Perceptron (MLP) model  
•	Evaluate performance using precision, recall, and F1-score  
•	Save the model and scaler for deployment  
•	Build a Streamlit app to allow real-time predictions  

### Libraries Used  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
```

### Dataset and Inputs  
The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, available directly from `sklearn.datasets.load_breast_cancer()`. It consists of 569 samples with 30 numeric features each, extracted from digitized breast mass images.

- Diagnosis classes:  
  • 0 — Malignant  
  • 1 — Benign  

- Features include:  
  • mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.  
  These represent various measurements of the cell nuclei in breast tissue.

### Model Summary  
- **Model used**: MLPClassifier (Multilayer Perceptron)  
- **Structure**: Two hidden layers with 64 and 32 neurons respectively  
- **Activation function**: ReLU  
- **Optimizer**: Adam  
- **Epochs**: 500  
- **Preprocessing**: StandardScaler used to normalize the features

### Evaluation Metrics  
The model was evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Sample Classification Report:
              precision    recall  f1-score   support

   malignant       0.96      0.94      0.95        43
      benign       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.96      0.97       114
weighted avg       0.97      0.97      0.97       114
```

### Streamlit App  
A Streamlit-based web interface allows users to:
- Input all 30 feature values manually
- Click the **Predict** button
- Instantly receive a classification: **Malignant** or **Benign**

### Instructions to Run  

1. **Install Dependencies**  
   Run:
   ```bash
   pip install streamlit scikit-learn joblib numpy pandas
   ```

2. **Train and Save the Model**  
   Run:
   ```bash
   python train_and_save_model.py
   ```

3. **Launch the Streamlit App**  
   Run:
   ```bash
   streamlit run app.py
   ```

### Improvements  
Possible future enhancements include:
• Hyperparameter tuning using GridSearchCV  
• Trying alternative models like Random Forest or SVC  
• Adding visualizations of prediction confidence  
• Deploying the Streamlit app on the cloud  


### Author  
Your Name: Adithisri Vishwanath
