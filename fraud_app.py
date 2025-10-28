import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model and Scaler ---
# CRITICAL: These must be in the same folder as this script!
try:
    model = joblib.load('final_fraud_model.joblib')
    scaler = joblib.load('fraud_scaler.joblib')
except FileNotFoundError:
    st.error("ERROR: Model or Scaler files not found. Please ensure 'final_fraud_model.joblib' and 'fraud_scaler.joblib' are in the same directory.")
    st.stop()


# --- 2. Streamlit UI Setup ---

st.set_page_config(layout="wide") 
st.title("💳 Credit Card Fraud Detection Application")
st.markdown("Enter the 30 features (Time, V1-V28, Amount) for a risk analysis.")
st.write("---")

# List of all 30 features (MUST match the order used during training!)
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Initialize dictionary for user input
input_data = {}

# --- 3. Input Fields ---

st.header("Enter Transaction Details")

# Arrange inputs in 3 columns for better layout
cols = st.columns(3)
col_index = 0

for feature in feature_names:
    with cols[col_index]:
        # Set realistic defaults 
        default_val = 0.0
        if feature == 'Time':
            default_val = 10000.0
        elif feature == 'Amount':
            default_val = 50.0

        # Create the number input box
        value = st.number_input(
            f"{feature}", 
            value=default_val, 
            format="%.6f", 
            key=feature,
            help="Value from the PCA-transformed dataset." if 'V' in feature else ""
        )
        input_data[feature] = value
        
    # Cycle column index: 0, 1, 2, 0, 1, 2...
    col_index = (col_index + 1) % 3


# --- 4. Prediction Logic ---

st.write("---")

if st.button("Analyze Transaction"):
    # 1. Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 2. Rescale the data using the loaded scaler (CRITICAL)
    input_df_scaled = scaler.transform(input_df) 
    
    # 3. Get Prediction (0=Normal, 1=Fraud) and Probabilities
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)[0]
    
    # 4. Display Result
    st.subheader("Analysis Result:")
    
    if prediction[0] == 0:
        st.success(f"✅ Prediction: **NORMAL TRANSACTION**")
        st.write(f"Confidence: **{prediction_proba[0]*100:.2f}%**")
    else:
        st.error(f"🚨 Prediction: **HIGH RISK OF FRAUD**")
        st.write(f"Confidence: **{prediction_proba[1]*100:.2f}%**")
        st.balloons()