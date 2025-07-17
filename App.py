import streamlit as st
import pandas as pd
import joblib 
import warnings
warnings.filterwarnings("ignore")


# Load the pre-trained model
model = joblib.load(r'KNN_heart.pkl')
# Import the scaler
scaler = joblib.load(r'scaler.pkl')
# import Columns
columns = joblib.load(r'columns.pkl')

# Create the UI
st.title("Heart Disease Prediction App❤️️")
st.markdown("Provide the following details to predict the risk of heart disease:")

# Columns : Age	RestingBP	Cholesterol	FastingBS	MaxHR	Oldpeak	HeartDisease	Sex_M	ChestPainType_ATA	ChestPainType_NAP	ChestPainType_TA	RestingECG_Normal	RestingECG_ST	ExerciseAngina_Y	ST_Slope_Flat	ST_Slope_Up

age = st.slider("Age", min_value=18, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "TA" , 'ASY'])

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0 , 1])
resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", options=["Y", "N"])
oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox("ST Slope", options=["Flat", "Up", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex : 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Create a DataFrame from the input
    input_data = pd.DataFrame([raw_input])

    # Fill missing columns with 0
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder the columns to match the model's expected input
    input_data = input_data[columns]
    # Scale 
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else :
        st.success("✅ Low Risk of Heart Disease")


