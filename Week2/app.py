import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
model = joblib.load('models/xgboost_model.pkl')
classifier = joblib.load('models/classifier2.pkl')

st.title("AI Retention Prediction System")

st.write(
    "Predict whether heavy AI usage affects "
    "long-term knowledge retention."
)

# User inputs

weekly_ai_hours = st.slider("Weekly AI Usage Hours",0,40,10)

study_hours = st.slider(
    "Traditional Study Hours",0,40,15)

pre_gpa = st.slider(
    "Pre Semester GPA",0.0,4.0,3.0)

post_gpa = st.slider("Post Semester GPA",0.0,4.0,3.2)

dependency = st.slider(
    "AI Dependency Level",1,10,5)

prompt_skill = st.slider(
    "Prompt Engineering Skill",1,10,5)

burnout = st.selectbox(
    "Burnout Risk",["Low", "Medium", "High"])

# Encode burnout
burnout_map = {
    "Low": 0,"Medium": 1, "High": 2}

burnout_encoded = burnout_map[burnout]

# Feature dataframe
input_df = pd.DataFrame({
    'Weekly_GenAI_Hours': [weekly_ai_hours],
    'Traditional_Study_Hours': [study_hours],
    'Pre_Semester_GPA': [pre_gpa],
    'Post_Semester_GPA': [post_gpa],
    'Perceived_AI_Dependency': [dependency],
    'Prompt_Engineering_Skill': [prompt_skill],
    'Burnout_Risk_Level': [burnout_encoded]
})

# PREDICTIONS

if st.button("Predict Retention"):

    retention_score = model.predict(input_df)[0]
    retention_class = classifier.predict(input_df)[0]
    st.subheader("Prediction Results")
    st.write(
        f"Predicted Retention Score: "
        f"{retention_score:.2f}"
    )

    if retention_class == 1:
        st.success("High Knowledge Retention")
    else:
        st.error("Low Knowledge Retention")

    # AI dependency insight
    if weekly_ai_hours >= 15:
        st.warning(
            "Heavy AI usage detected. "
            "Long-term retention may decrease."
        )