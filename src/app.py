from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from visualization import show_visualizations


def load_model_and_data():
    """Load model and data with proper error handling"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.pkl')
        data_path = os.path.join(
            current_dir, '..', 'data', 'processed_data.csv')

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        return model, df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.error(
            "Make sure `model.pkl` is in `src/` and `processed_data.csv` is in `data/`.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data/model: {e}")
        st.stop()


# Load model and data
model, df = load_model_and_data()

# Load original data to create mappings
current_dir = os.path.dirname(os.path.abspath(__file__))
original_data_path = os.path.join(
    current_dir, '..', 'data', 'cleaned_data.csv')
df_original = pd.read_csv(original_data_path)

# Encoders
le_region = LabelEncoder()
le_property_type = LabelEncoder()

le_region.fit(df_original['addressRegion'].dropna().unique())
le_property_type.fit(df_original['Property Type'].dropna().unique())


# Sidebar pages
page = st.sidebar.selectbox("Choose Page", ["Predict Price", "Visualizations"])


if page == "Predict Price":
    st.header("🏠 Property Price Predictor")
    st.markdown(
        "Enter your property details below to get an estimated price prediction.")

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏡 Property Details")
        beds = st.number_input("Number of bedrooms", 1, 10, 3)
        baths = st.number_input("Number of bathrooms", 1, 10, 2)
        sqft = st.number_input("Square footage", 500, 10000, 1500)
        acreage = st.number_input("Acreage", 0.0, 10.0, 0.5, step=0.1)

    with col2:
        st.subheader("📍 Location Details")
        lat = st.number_input("Latitude", 40.0, 70.0, 45.0, step=0.1)
        lon = st.number_input("Longitude", -140.0, -50.0, -79.0, step=0.1)
        region = st.selectbox(
            "Province/Region", df_original['addressRegion'].unique())
        property_type = st.selectbox(
            "Property Type", df_original['Property Type'].unique())

    st.subheader("✨ Additional Features")
    col3, col4 = st.columns(2)

    with col3:
        has_fireplace = st.checkbox("Has Fireplace", False)
        basement = st.checkbox("Has Basement", False)

    with col4:
        has_garage = st.checkbox("Has Garage", False)
        heating = st.checkbox("Has Heating", False)
        parking = st.checkbox("Has Parking", False)

    st.info("💡 Tip: More bedrooms, bathrooms, and square footage usually increase property value. Location and features also play key roles.")

    # Prediction
    st.subheader("🎯 Price Prediction")

    if st.button("🔮 Get Price Prediction", type="primary"):
        if sqft <= 0 or beds < 1 or baths < 1 or acreage < 0:
            st.error("Invalid input values. Please check your entries.")
        elif lat < 40.0 or lat > 70.0 or lon < -140.0 or lon > -50.0:
            st.error(
                "Coordinates must be within Canada (lat: 40–70, lon: -140–-50).")
        else:
            with st.spinner("Calculating price prediction..."):
                try:
                    # Haversine distance
                    def haversine(lat1, lon1, lat2, lon2):
                        R = 6371
                        dlat = np.radians(lat2 - lat1)
                        dlon = np.radians(lon2 - lon1)
                        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
                            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                        c = 2 * np.arcsin(np.sqrt(a))
                        return R * c

                    dist_to_toronto = haversine(lat, lon, 43.7, -79.4)

                    # Adjust price per sqft
                    base_ppsqft = df['price_per_sqft'].median()
                    median_sqft = df['Square Footage'].median()
                    adjustment_factor = (
                        sqft / median_sqft if median_sqft > 0 else 1) * (1 + 0.1 * (beds + baths) / 10)
                    adjusted_ppsqft = base_ppsqft * adjustment_factor

                    # Encode categorical features
                    encoded_region = int(le_region.transform([region])[0])
                    encoded_property_type = int(
                        le_property_type.transform([property_type])[0])

                    # Match training features
                    feature_columns = [
                        'property-beds', 'property-baths', 'Square Footage', 'Acreage', 'latitude', 'longitude',
                        'price_per_sqft', 'has_fireplace', 'has_garage', 'dist_to_toronto_km',
                        'addressRegion', 'Property Type', 'Basement', 'Fireplace', 'Heating', 'Parking'
                    ]

                    input_data = {
                        'property-beds': beds,
                        'property-baths': baths,
                        'Square Footage': sqft,
                        'Acreage': acreage,
                        'latitude': lat,
                        'longitude': lon,
                        'price_per_sqft': adjusted_ppsqft,
                        'has_fireplace': int(has_fireplace),
                        'has_garage': int(has_garage),
                        'dist_to_toronto_km': dist_to_toronto,
                        'addressRegion': encoded_region,
                        'Property Type': encoded_property_type,
                        'Basement': int(basement),
                        'Fireplace': int(has_fireplace),
                        'Heating': int(heating),
                        'Parking': int(parking)
                    }

                    input_df = pd.DataFrame([input_data])[feature_columns]

                    # Predict
                    pred = model.predict(input_df)[0]

                    st.success(f"💰 Estimated Price: ${pred:,.2f} CAD")

                    # Insights
                    col5, col6, col7 = st.columns(3)

                    with col5:
                        price_per_sqft = pred / sqft
                        st.metric("Price per Sq Ft", f"${price_per_sqft:,.2f}")

                    with col6:
                        avg_price = df['price'].mean()
                        price_diff = pred - avg_price
                        st.metric(
                            "vs Market Average", f"${price_diff:,.2f}", delta=f"{price_diff/avg_price*100:.1f}%")

                    with col7:
                        st.metric("Distance to Toronto",
                                  f"{dist_to_toronto:.1f} km")

                    price_percentile = (df['price'] <= pred).mean() * 100
                    st.info(
                        f"This property is in the **{price_percentile:.0f}th percentile** of the market.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")


elif page == "Visualizations":
    show_visualizations()
