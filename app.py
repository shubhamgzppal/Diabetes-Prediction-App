import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load all models and scaler from single pickle file
with open("diabetes_models.pkl", "rb") as f:
    saved_objects = pickle.load(f)

# Extract scaler
scaler = saved_objects["Scaler"]

# Extract models (everything except scaler)
models = {k: v for k, v in saved_objects.items() if k != "Scaler"}

# Streamlit UI setup
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient data below to predict the likelihood of diabetes using multiple ML models.")

# Input form for user data
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Female pregnant time.(for male it will be 0)", min_value=0, step=1)
        Glucose = st.number_input("Glucose : Plasma glucose concentration (in mg/dL", min_value=0.0)
        BloodPressure = st.number_input("Blood Pressure : Diastolic blood pressure (in mm Hg)", min_value=0.0)
        SkinThickness = st.number_input("Skin Thickness (Measures the thickness of the triceps skinfold in millimeters)", min_value=0.0)

    with col2:
        Insulin = st.number_input("Insulin: Serum insulin level (in µU/mL)", min_value=0.0)
        BMI = st.number_input("Body Mass Index", min_value=0.0)
        DiabetesPedigreeFunction = st.number_input("Genetic background Risk (Family History with Diabities)", min_value=0.0)
        Age = st.number_input("Age", min_value=1, step=1)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_scaled = scaler.transform(input_data)

    st.subheader("🔬 Predictions by Model")
    diabetes_votes = 0

    for model_name, model in models.items():
        pred = model.predict(input_scaled)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0][1]
        else:
            # Some models like SVM might not have predict_proba; fallback with decision_function
            try:
                decision = model.decision_function(input_scaled)[0]
                proba = 1 / (1 + np.exp(-decision))  # sigmoid to convert to probability
            except:
                proba = None

        prediction_text = "🔴 Diabetes" if pred == 1 else "🟢 No Diabetes"

        if pred == 1:
            diabetes_votes += 1

        if proba is not None:
            st.markdown(f"**{model_name}**: {prediction_text} (Probability: {proba:.4f})")
        else:
            st.markdown(f"**{model_name}**: {prediction_text}")

    st.markdown("---")
    st.subheader("📊 Final Verdict")
    st.markdown(f"**{diabetes_votes} out of {len(models)} models** predict **🔴 Diabetes**")

    if diabetes_votes >= len(models) / 2:
        st.success("✅ Likely Diabetic")
    else:
        st.info("✅ Likely Not Diabetic")
