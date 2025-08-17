import streamlit as st
import pandas as pd
import joblib
import os

# -------------------
# Load Model + Scaler
# -------------------
MODEL_PATH = "ticket_booking_model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# -------------------
# Label Encoding Maps (same as training encoders)
# -------------------
sales_channel_map = {"Internet": 0, "Mobile": 1, "Agent": 2}
trip_type_map = {"RoundTrip": 0, "OneWay": 1}
flight_day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

# ⚠️ These must match your training encoders!
# You may need to adjust them to match your original LabelEncoder fitting.
# For now we’ll just hash unseen categories into -1.
def safe_map(mapping, key):
    return mapping.get(key, -1)

# -------------------
# Streamlit UI
# -------------------
st.title("✈️ Flight Booking Prediction App")
st.write("This app predicts whether a customer will complete a booking or not.")

st.header("Enter Booking Details")

num_passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
sales_channel = st.selectbox("Sales Channel", list(sales_channel_map.keys()))
trip_type = st.selectbox("Trip Type", list(trip_type_map.keys()))
purchase_lead = st.slider("Purchase Lead (days before travel)", 0, 365, 15)
length_of_stay = st.slider("Length of Stay (days)", 1, 365, 7)
flight_hour = st.slider("Flight Hour (0-23)", 0, 23, 17)
flight_day = st.selectbox("Flight Day", list(flight_day_map.keys()))
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
    "sales_channel": safe_map(sales_channel_map, sales_channel),
    "trip_type": safe_map(trip_type_map, trip_type),
    "purchase_lead": purchase_lead,
    "length_of_stay": length_of_stay,
    "flight_hour": flight_hour,
    "flight_day": safe_map(flight_day_map, flight_day),
    # Route + booking_origin must be mapped properly
    # Placeholder: simple hash to int (should match training encoder)
    "route": hash(route) % 1000,
    "booking_origin": hash(booking_origin) % 1000,
    "wants_extra_baggage": 1 if wants_extra_baggage == "Yes" else 0,
    "wants_preferred_seat": 1 if wants_preferred_seat == "Yes" else 0,
    "wants_in_flight_meals": 1 if wants_in_flight_meals == "Yes" else 0,
    "flight_duration": flight_duration,
}

input_df = pd.DataFrame([input_dict])

# Apply scaler
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error scaling input: {e}")
    st.stop()

# -------------------
# Prediction
# -------------------
if st.button("Predict Booking"):
    proba = model.predict_proba(input_scaled)[0, 1]
    prediction = 1 if proba >= 0.4 else 0
    st.subheader("🔮 Prediction Result")
    st.write(f"**Booking Probability:** {proba:.2f}")

    if prediction == 1:
        st.success("✅ This customer is likely to complete the booking.")
    else:
        st.error("❌ This customer may not complete the booking.")
