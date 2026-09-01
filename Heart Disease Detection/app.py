import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("model_knn.pkl")
scalar = joblib.load("scalar.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction by Yash ❤️")
st.markdown("Provide the following details")

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80, 200, 120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    100, 600, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Max Heart Rate",
    60, 220, 150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# Prediction
if st.button("Predict"):

    # Start with all expected columns as 0
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=expected_columns
    )

    # Numerical values
    input_data["Age"] = age
    input_data["RestingBP"] = resting_bp
    input_data["Cholesterol"] = cholesterol
    input_data["FastingBS"] = fasting_bs
    input_data["MaxHR"] = max_hr
    input_data["Oldpeak"] = oldpeak

    # Categorical values
    if sex == "M":
        input_data["Sex_M"] = 1

    if chest_pain == "ATA":
        input_data["ChestPainType_ATA"] = 1
    elif chest_pain == "NAP":
        input_data["ChestPainType_NAP"] = 1
    elif chest_pain == "TA":
        input_data["ChestPainType_TA"] = 1

    if resting_ecg == "Normal":
        input_data["RestingECG_Normal"] = 1
    elif resting_ecg == "ST":
        input_data["RestingECG_ST"] = 1

    if exercise_angina == "Y":
        input_data["ExerciseAngina_Y"] = 1

    if st_slope == "Flat":
        input_data["ST_Slope_Flat"] = 1
    elif st_slope == "Up":
        input_data["ST_Slope_Up"] = 1

    # Scale numerical columns
    scale_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    input_data[scale_cols] = scalar.transform(
        input_data[scale_cols]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")