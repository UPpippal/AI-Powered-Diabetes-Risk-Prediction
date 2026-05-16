import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(
    "model/diabetes_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title(
    "🩺 Diabetes Risk Prediction System"
)

st.markdown(
    "Enter patient health details below"
)

st.divider()

# Inputs
BMI = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=100.0,
    value=25.0
)

Age = st.slider(
    "Age Category (1–13)",
    1,
    13,
    5
)

HighBP = st.selectbox(
    "High Blood Pressure",
    [0, 1]
)

HighChol = st.selectbox(
    "High Cholesterol",
    [0, 1]
)

PhysHlth = st.slider(
    "Physical Health Days (0–30)",
    0,
    30,
    0
)

GenHlth = st.slider(
    "General Health (1=Excellent, 5=Poor)",
    1,
    5,
    3
)

# Predict button
if st.button("Predict Diabetes Risk"):

    # 21 features matching training data
    input_data = np.array([[
        HighBP,      # HighBP
        HighChol,    # HighChol
        1,            # CholCheck
        BMI,          # BMI
        0,            # Smoker
        0,            # Stroke
        0,            # HeartDiseaseorAttack
        1,            # PhysActivity
        1,            # Fruits
        1,            # Veggies
        0,            # HvyAlcoholConsump
        1,            # AnyHealthcare
        0,            # NoDocbcCost
        GenHlth,      # GenHlth
        0,            # MentHlth
        PhysHlth,     # PhysHlth
        0,            # DiffWalk
        1,            # Sex
        Age,          # Age
        4,            # Education
        5             # Income
    ]])

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ High Diabetes Risk ({probability:.2%})"
        )
    else:
        st.success(
            f"✅ Low Diabetes Risk ({probability:.2%})"
        )

    st.info(
        "Disclaimer: This tool is for educational purposes only and not medical advice."
    )
    st.divider()

st.subheader(
    "🤖 AI Health Assistant"
)

user_question = st.text_input(
    "Ask health-related questions"
)

if st.button("Ask AI"):

    from chatbot import get_health_advice

    response = get_health_advice(
        user_question
    )

    st.success(response)