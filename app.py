import streamlit as st
import joblib
import numpy as np
from groq import Groq

# --- Load model and encoders ---
model_bundle = joblib.load("crime_prediction_model.pkl")
model = model_bundle["model"]
type_encoding_map = model_bundle["type_encoding_map"]
season_encoding_map = model_bundle["season_encoding_map"]
crime_era_encoding_map = model_bundle["crime_era_encoding_map"]
neighbourhood_classes = model_bundle["neighbourhood_classes"]

# --- Groq client ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- Page config ---
st.set_page_config(page_title="FBI Crime Predictor", layout="centered")
st.title("FBI Crime Investigation — Crime Count Predictor")
st.markdown("Fill in the details below to predict expected crime count and get an AI-generated insight.")

# --- User Inputs ---
neighbourhood = st.selectbox("Neighbourhood", options=neighbourhood_classes)
crime_type = st.selectbox("Crime Type", options=list(type_encoding_map.keys()))
season = st.selectbox("Season", options=list(season_encoding_map.keys()))
crime_era = st.selectbox("Crime Era", options=list(crime_era_encoding_map.keys()))

# --- Predict button ---
if st.button("Predict Crime Count"):

    # Encode inputs
    neighbourhood_enc = neighbourhood_classes.index(neighbourhood)
    type_enc = type_encoding_map[crime_type]
    season_enc = season_encoding_map[season]
    era_enc = crime_era_encoding_map[crime_era]

    # Build feature array
    features = np.array([[neighbourhood_enc, type_enc, season_enc, era_enc]])

    # Predict and reverse log transform
    prediction_log = model.predict(features)[0]
    prediction = int(np.expm1(prediction_log))

    st.success(f"Predicted Crime Count: **{prediction:,}**")

    # --- Groq Narrative ---
    with st.spinner("Generating AI insight..."):
        prompt = f"""
        You are a crime analyst assistant. A prediction model estimates that in the 
        '{neighbourhood}' neighbourhood, during {season} in the era {crime_era}, 
        the crime type '{crime_type}' is expected to occur approximately {prediction:,} times.
        
        Write a short 5-sentence insight for a policy maker explaining what this means, 
        what might be driving it, and one actionable recommendation.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        narrative = response.choices[0].message.content
        st.markdown("### AI Insight")
        st.write(narrative)