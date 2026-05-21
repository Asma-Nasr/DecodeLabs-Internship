import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

with open('Week3/models/label_encoder.json', 'r') as f:
    classes = json.load(f)
le = LabelEncoder()
le.classes_ = np.array(classes)

# XGBoost Model
classifier = XGBClassifier()
classifier.load_model("Week3/models/xgboost_model.json")

# Random Forest Model
rf_model = joblib.load('Week3/models/random_forest_model.joblib')

st.set_page_config(
    page_title="Sleep Disorder Detection",page_icon="😴",layout="centered")
st.title("😴 Sleep Disorder Detection App")
st.write(
    "Enter patient health and lifestyle information "
    "to predict possible sleep disorders.")
# User Inputs
gender = st.selectbox("Gender",["Male", "Female"])
age = st.number_input("Age",min_value=1,max_value=100,value=25)
occupation = st.selectbox("Occupation",[
        "Doctor", "Engineer", "Teacher", "Nurse", "Lawyer", "Manager",
        "Salesperson","Scientist","Accountant","Student","Other"])

sleep_duration = st.slider("Sleep Duration (hours)",0.0,12.0,7.0)
quality_sleep = st.slider("Quality of Sleep",1,10,5)
physical_activity = st.slider("Physical Activity Level",0,100,50)
stress_level = st.slider("Stress Level",1,10,5)
bmi_category = st.selectbox("BMI Category",["Normal", "Normal Weight", "Overweight", "Obese"])
heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200,value=70)
daily_steps = st.number_input("Daily Steps",min_value=0,max_value=50000,value=5000)
systolic_bp = st.number_input("Systolic BP",min_value=70,max_value=250,value=120)
diastolic_bp = st.number_input("Diastolic BP",min_value=40,max_value=150,value=80)

# Encoding Maps
gender_map = { "Male": 1,"Female": 0}

occupation_map = {
    "Doctor": 0,"Engineer": 1, "Teacher": 2,"Nurse": 3, "Lawyer": 4,"Manager": 5, "Salesperson": 6,"Scientist": 7, "Accountant": 8,"Student": 9,"Other": 10}

bmi_map = {
    "Normal": 0,"Normal Weight": 1,"Overweight": 2,"Obese": 3}

# Create Input DataFrame
input_data = pd.DataFrame({
    'Gender': [gender_map[gender]],
    'Age': [age],
    'Occupation': [occupation_map[occupation]],
    'Sleep Duration': [sleep_duration],
    'Quality of Sleep': [quality_sleep],
    'Physical Activity Level': [physical_activity],
    'Stress Level': [stress_level],
    'BMI Category': [bmi_map[bmi_category]],
    'Heart Rate': [heart_rate],
    'Daily Steps': [daily_steps],
    'Systolic_BP': [systolic_bp],
    'Diastolic_BP': [diastolic_bp] })
# Prediction Section
model_choice = st.selectbox(
    "Choose Model",
    ["XGBoost", "Random Forest"])
if st.button("Predict"):
    # Select model
    if model_choice == "XGBoost":
        model_used = classifier
    else:
        model_used = rf_model

    # Prediction
    prediction = model_used.predict(input_data)
    # Confidence scores
    probabilities = model_used.predict_proba(input_data)
    # Highest confidence
    confidence = np.max(probabilities) * 100

    # Decode label
    prediction_label = le.inverse_transform(
        prediction.astype(int))
    if prediction_label[0] == "Sleep Apnea":
        st.error(
            f"⚠️ Potential Sleep Apnea Detected\n\n"
            f"Confidence: {confidence:.2f}%")
    elif prediction_label[0] == "Insomnia":
        st.warning(
            f"⚠️ Possible Insomnia Detected\n\n"
            f"Confidence: {confidence:.2f}%")
    else:
        st.success(
            f"✅ No Sleep Disorder Detected\n\n"
            f"Confidence: {confidence:.2f}%")
