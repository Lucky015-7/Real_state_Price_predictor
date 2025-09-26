import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🏠 Real Estate Price Predictor (Canada)")

# User inputs
beds = st.number_input("Number of Bedrooms", 1, 10, 3)
sqft = st.number_input("Square Footage", 500, 10000, 1000)
lat = st.number_input("Latitude", 40.0, 70.0, 45.0)
lon = st.number_input("Longitude", -140.0, -50.0, -79.0)

# Predict button
if st.button("Predict Price"):
    X_new = pd.DataFrame([[beds, sqft, lat, lon]],
                         columns=["property-beds", "Square Footage", "latitude", "longitude"])
    price_pred = model.predict(X_new)[0]
    st.success(f"Estimated Price: ${price_pred:,.2f}")
