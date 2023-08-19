import streamlit as st
import joblib
import numpy as np

# Load the saved pipeline
loaded_pipeline = joblib.load('insurance_model_pipeline.joblib')

# Streamlit app interface
st.title("Insurance Cost Prediction App")

# Input fields for user to provide input features
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10, max_value=50, value=25)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Feature mappings (similar to previous examples)
sex_mapping = {"Male": 0, "Female": 1}
smoker_mapping = {"Yes": 1, "No": 0}
region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

sex_encoded = sex_mapping[sex]
smoker_encoded = smoker_mapping[smoker]
region_encoded = region_mapping[region]

# Predict button
if st.button("Predict"):
    # Create a feature vector using user inputs
    features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    prediction = loaded_pipeline.predict(features)[0]

    st.write(f"Predicted Insurance Cost: ${prediction:.2f}")
