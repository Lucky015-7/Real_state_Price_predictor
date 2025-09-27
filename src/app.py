from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from visualization import show_visualizations

# Load model + data with proper path handling


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
        st.error("Please make sure the model.pkl file exists in the src directory and processed_data.csv exists in the data directory.")
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

# Create encoders and fit them on the original data to match the UI input
le_region = LabelEncoder()
le_property_type = LabelEncoder()

# Fit encoders on original data (from cleaned_data.csv) to match the selectbox options
le_region.fit(df_original['addressRegion'].dropna().unique())
le_property_type.fit(df_original['Property Type'].dropna().unique())

# Debug: Show available categories
st.write("Debug - Available Regions in Original Data:",
         df_original['addressRegion'].dropna().unique())
st.write("Debug - Encoded Regions:",
         dict(zip(le_region.classes_, le_region.transform(le_region.classes_))))
st.write("Debug - Available Property Types in Original Data:",
         df_original['Property Type'].dropna().unique())

# Pages
page = st.sidebar.selectbox(
    "Choose Page", ["Predict Price", "Upload & Analyze Data", "Visualizations"])

if page == "Predict Price":
    st.header("🏠 Property Price Predictor")
    st.markdown(
        "Enter your property details below to get an estimated price prediction.")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏡 Property Details")
        beds = st.number_input("Number of bedrooms", 1,
                               10, 3, help="Enter the number of bedrooms")
        baths = st.number_input("Number of bathrooms",
                                1, 10, 2, help="Enter the number of bathrooms")
        sqft = st.number_input("Square footage", 500, 10000,
                               1500, help="Enter the total square footage")
        acreage = st.number_input(
            "Acreage", 0.0, 10.0, 0.5, step=0.1, help="Enter the lot size in acres")

    with col2:
        st.subheader("📍 Location Details")
        lat = st.number_input("Latitude", 40.0, 70.0, 45.0,
                              step=0.1, help="Enter the latitude coordinate")
        lon = st.number_input("Longitude", -140.0, -50.0, -79.0,
                              step=0.1, help="Enter the longitude coordinate")
        region = st.selectbox(
            "Province/Region", df_original['addressRegion'].unique(), help="Select the province or region")
        property_type = st.selectbox(
            "Property Type", df_original['Property Type'].unique(), help="Select the property type")

    # Features section
    st.subheader("✨ Additional Features")
    col3, col4 = st.columns(2)

    with col3:
        has_fireplace = st.checkbox(
            "Has Fireplace", False, help="Check if the property has a fireplace")
        basement = st.checkbox("Has Basement", False,
                               help="Check if the property has a basement")

    with col4:
        has_garage = st.checkbox(
            "Has Garage", False, help="Check if the property has a garage")
        heating = st.checkbox("Has Heating", False,
                              help="Check if the property has heating")
        parking = st.checkbox("Has Parking", False,
                              help="Check if the property has parking")

    st.info("💡 **Tip**: More bedrooms, bathrooms, and square footage typically increase property value. Location and additional features also play important roles.")

    # Prediction section
st.subheader("🎯 Price Prediction")

if st.button("🔮 Get Price Prediction", type="primary"):
    # Input validation
    if sqft <= 0:
        st.error("Square footage must be greater than 0")
    elif beds <= 0 or baths <= 0:
        st.error("Number of bedrooms and bathrooms must be greater than 0")
    else:
        with st.spinner("Calculating price prediction..."):
            try:
                # Compute engineered features
                def haversine(lat1, lon1, lat2, lon2):
                    """Calculate distance between two points using Haversine formula"""
                    R = 6371  # Earth's radius in kilometers
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
                        np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    return R * c

                dist_to_toronto = haversine(lat, lon, 43.7, -79.4)

                # Estimate price_per_sqft dynamically with added variation
                base_ppsqft = df['price_per_sqft'].median()
                median_sqft = df['Square Footage'].median()
                adjustment_factor = (
                    sqft / median_sqft if median_sqft > 0 else 1) * (1 + 0.1 * (beds + baths) / 10)
                adjusted_ppsqft = base_ppsqft * adjustment_factor

                # Convert selected string values to encoded numeric values
                encoded_region = le_region.transform([region])[0]
                if not np.issubdtype(type(encoded_region), np.integer):
                    st.error(
                        f"Encoding error: Region '{region}' transformed to {encoded_region} (type: {type(encoded_region)}), expected integer.")
                    st.stop()
                encoded_region = int(encoded_region)
                encoded_property_type = le_property_type.transform([property_type])[
                    0]
                if not np.issubdtype(type(encoded_property_type), np.integer):
                    st.error(
                        f"Encoding error: Property Type '{property_type}' transformed to {encoded_property_type} (type: {type(encoded_property_type)}), expected integer.")
                    st.stop()
                encoded_property_type = int(encoded_property_type)

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

                # Debugging: Print input data and model type
                st.write("Input Data:", input_data)
                st.write("Input DataFrame:", input_df)
                st.write("Model Type:", model.__class__.__name__)

                # Predict
                pred = model.predict(input_df)[0]

                # Add timestamp to verify each prediction is unique
                import time
                timestamp = time.time()

                # Display results
                st.success(f"💰 **Estimated Price: ${pred:,.2f} CAD**")
                st.write(f"🕐 **Prediction Time**: {timestamp}")

                # Quick debug to verify inputs are changing
                st.write(
                    f"🔍 **Input Summary**: {beds} bed, {baths} bath, {sqft} sqft, {region} ({encoded_region}), {property_type} ({encoded_property_type})")

                # Show the actual input values being sent to model
                st.write("📊 **Model Input Values**:")
                st.write(f"- Bedrooms: {beds} (type: {type(beds)})")
                st.write(f"- Bathrooms: {baths} (type: {type(baths)})")
                st.write(f"- Square Footage: {sqft} (type: {type(sqft)})")
                st.write(
                    f"- Region: {region} → {encoded_region} (type: {type(encoded_region)})")
                st.write(
                    f"- Property Type: {property_type} → {encoded_property_type} (type: {type(encoded_property_type)})")

                # Additional insights
                st.subheader("📊 Price Analysis")

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

                # Price range comparison
                st.subheader("📈 Market Context")
                price_percentile = (df['price'] <= pred).mean() * 100
                st.info(
                    f"This property is estimated to be in the **{price_percentile:.0f}th percentile** of the market.")

                # Optional: Display feature importances if available
                if hasattr(model, 'feature_importances_'):
                    st.write("Feature Importances:", dict(
                        zip(feature_columns, model.feature_importances_)))

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.error("Please check your input values and try again.")
