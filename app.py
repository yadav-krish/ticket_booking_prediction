import streamlit as st
import pandas as pd
import joblib
import json
import requests
import os

# -------------------
# Download & Load Model
# -------------------
MODEL_PATH = "random_forest_booking_pipeline.pkl"
# Replace this with your hosted model URL (Google Drive / S3)
MODEL_URL = "https://drive.google.com/file/d/1LsyO9KETxdsI4OZg54LdqkteS2NqEpnx/view?usp=drive_link"

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading large model, please wait...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded!")

    # Load model
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# -------------------
# Load Threshold
# -------------------
try:
    with open("best_threshold.json", "r") as f:
        best_threshold = json.load(f)["threshold"]
except:
    best_threshold = 0.5  # fallback if file missing

# -------------------
# Streamlit UI
# -------------------
st.title("✈️ Flight Booking Prediction App")
st.write("This app predicts whether a customer will complete a booking or not.")

st.header("Enter Booking Details")

# Collect inputs
num_passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
sales_channel = st.selectbox("Sales Channel", ["Internet", "Mobile", "Agent"])
trip_type = st.selectbox("Trip Type", ["RoundTrip", "OneWay"])
purchase_lead = st.slider("Purchase Lead (days before travel)", 0, 365, 15)
length_of_stay = st.slider("Length of Stay (days)", 1, 365, 7)
flight_hour = st.slider("Flight Hour (0-23)", 0, 23, 17)
flight_day = st.selectbox("Flight Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
route = st.text_input("Route (e.g. AKLKUL)", "AKLKUL")
booking_origin = st.text_input("Booking Origin (e.g. India, Malaysia)", "India")

wants_extra_baggage = st.selectbox("Wants Extra Baggage?", ["No", "Yes"])
wants_preferred_seat = st.selectbox("Wants Preferred Seat?", ["No", "Yes"])
wants_in_flight_meals = st.selectbox("Wants In-Flight Meals?", ["No", "Yes"])
flight_duration = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=24.0, value=8.0)

# -------------------
# Prepare Input Data
# -------------------
input_dict = {
    "num_passengers": num_passengers,
    "sales_channel": sales_channel,
    "trip_type": trip_type,
    "purchase_lead": purchase_lead,
    "length_of_stay": length_of_stay,
    "flight_hour": flight_hour,
    "flight_day": flight_day,
    "route": route,
    "booking_origin": booking_origin,
    "wants_extra_baggage": 1 if wants_extra_baggage == "Yes" else 0,
    "wants_preferred_seat": 1 if wants_preferred_seat == "Yes" else 0,
    "wants_in_flight_meals": 1 if wants_in_flight_meals == "Yes" else 0,
    "flight_duration": flight_duration
}

input_df = pd.DataFrame([input_dict])

# -------------------
# Prediction
# -------------------
if st.button("Predict Booking"):
    proba = model.predict_proba(input_df)[0, 1]
    prediction = 1 if proba >= best_threshold else 0

    st.subheader("🔮 Prediction Result")
    st.write(f"**Booking Probability:** {proba:.2f}")

    if prediction == 1:
        st.success("✅ This customer is likely to complete the booking.")
    else:
        st.error("❌ This customer may not complete the booking.")
