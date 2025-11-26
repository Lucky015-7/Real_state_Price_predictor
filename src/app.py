from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from visualization import show_visualizations

# ────────────────────────── THE BUG BUSTERS — EPIC UI UPGRADE ──────────────────────────
st.set_page_config(
    page_title="The Bug Busters | Real Estate Price Predictor",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — makes it look like a $10,000 app
st.markdown("""
<style>
    .main {background-color: #0e1117; color: #fafafa;}
    .stButton>button {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        color: white; font-weight: bold; border: none; border-radius: 12px;
        padding: 12px 24px; height: 60px; font-size: 18px;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #262730; color: white; border-radius: 10px;
    }
    h1, h2, h3 {font-family: 'Montserrat', sans-serif; color: #feca57;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# EPIC TITLE
st.markdown("""
<div style="text-align: center; padding: 30px;">
    <h2 style="color: #feca57;"> Real Estate Price Predictor</h2>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────────────


def load_model_and_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.pkl')
        data_path = os.path.join(
            current_dir, '..', 'data', 'processed_data.csv')

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        return model, df
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        st.stop()


# Load model and processed data
model, df = load_model_and_data()

# Load original clean data for dropdown options only
current_dir = os.path.dirname(os.path.abspath(__file__))
original_data_path = os.path.join(
    current_dir, '..', 'data', 'cleaned_data.csv')
df_original = pd.read_csv(original_data_path)

# Fit encoders on original data
le_region = LabelEncoder().fit(df_original['addressRegion'].dropna().unique())
le_property_type = LabelEncoder().fit(
    df_original['Property Type'].dropna().unique())

# ===================================================================
# NEW: Pre-compute region → median price mapping (exactly like training)
# ===================================================================
region_to_median = df.groupby('addressRegion')[
    'region_median_price'].median().to_dict()
overall_median_price = df['region_median_price'].median()

# ===================================================================

# Sidebar
page = st.sidebar.selectbox("Choose Page", ["Predict Price", "Visualizations"])

if page == "Predict Price":
    st.header("Property Price Predictor")
    st.markdown("Enter property details to get an accurate price estimate.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Property Details")
        beds = st.number_input("Bedrooms", 1, 10, 3)
        baths = st.number_input("Bathrooms", 1, 10, 2)
        sqft = st.number_input("Square Footage", 500, 15000, 1500)
        acreage = st.number_input("Acreage", 0.0, 50.0, 0.5, step=0.1)

    with col2:
        st.subheader("Location Details")
        lat = st.number_input("Latitude", 40.0, 70.0, 45.0, step=0.01)
        lon = st.number_input("Longitude", -140.0, -50.0, -79.0, step=0.01)
        region = st.selectbox(
            "Province/Region", df_original['addressRegion'].dropna().unique())
        property_type = st.selectbox(
            "Property Type", df_original['Property Type'].dropna().unique())

    st.subheader("Additional Features")
    col3, col4 = st.columns(2)
    with col3:
        has_fireplace = st.checkbox("Has Fireplace", False)
        basement = st.checkbox("Has Basement", False)
    with col4:
        has_garage = st.checkbox("Has Garage", False)
        heating = st.checkbox("Has Heating", True)
        parking = st.checkbox("Has Parking", True)

    st.info(
        "Tip: Larger homes, more bedrooms/bathrooms, and better location increase value.")

    if st.button("Get Price Prediction", type="primary"):
        # Basic validation
        if sqft <= 0 or beds < 1 or baths < 1:
            st.error(
                "Please check your inputs – sqft, beds, baths must be positive.")
        else:
            with st.spinner("Predicting..."):
                try:
                    # Haversine distance to Toronto
                    def haversine(lat1, lon1, lat2=43.7, lon2=-79.4):
                        R = 6371
                        dlat = np.radians(lat2 - lat1)
                        dlon = np.radians(lon2 - lon1)
                        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
                            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                        c = 2 * np.arcsin(np.sqrt(a))
                        return R * c

                    dist_to_toronto = haversine(lat, lon)

                    # Encode categorical variables
                    encoded_region = int(le_region.transform([region])[0])
                    encoded_property_type = int(
                        le_property_type.transform([property_type])[0])

                    # GET region_median_price using the pre-computed mapping
                    region_median_price = region_to_median.get(
                        encoded_region, overall_median_price)

                    # FINAL CLEAN FEATURE LIST — NOW INCLUDES region_median_price!
                    feature_columns = [
                        'property-beds', 'property-baths', 'Square Footage', 'Acreage',
                        'latitude', 'longitude',
                        'has_fireplace', 'has_garage', 'dist_to_toronto_km',
                        'addressRegion', 'Property Type', 'Basement',
                        'Fireplace', 'Heating', 'Parking', 'region_median_price'
                    ]

                    input_data = {
                        'property-beds': beds,
                        'property-baths': baths,
                        'Square Footage': sqft,
                        'Acreage': acreage,
                        'latitude': lat,
                        'longitude': lon,
                        'has_fireplace': int(has_fireplace),
                        'has_garage': int(has_garage),
                        'dist_to_toronto_km': dist_to_toronto,
                        'addressRegion': encoded_region,
                        'Property Type': encoded_property_type,
                        'Basement': int(basement),
                        'Fireplace': int(has_fireplace),
                        'Heating': int(heating),
                        'Parking': int(parking),
                        'region_median_price': region_median_price   # ← CRITICAL FIX
                    }

                    input_df = pd.DataFrame([input_data])[feature_columns]

                    # Predict and reverse log transform (model was trained on log(price))
                    log_prediction = model.predict(input_df)[0]
                    prediction = np.expm1(log_prediction)   # ← CRITICAL FIX

                    st.success(f"Estimated Price: **${prediction:,.0f} CAD**")

                    # Extra insights
                    colA, colB, colC = st.columns(3)
                    with colA:
                        st.metric("Price per Sq Ft",
                                  f"${prediction/sqft:,.0f}")
                    with colB:
                        diff = prediction - df['price'].mean()
                        st.metric(
                            "vs Market Avg", f"${diff:,.0f}", delta=f"{diff/df['price'].mean()*100:.1f}%")
                    with colC:
                        st.metric("Distance to Toronto",
                                  f"{dist_to_toronto:.0f} km")

                    percentile = (df['price'] <= prediction).mean() * 100
                    st.info(
                        f"This property is in the **{percentile:.0f}th percentile** of the market.")

                except Exception as e:
                    st.error(f"Prediction error: {e}")

elif page == "Visualizations":
    show_visualizations()

# Footer
st.markdown("---")
st.markdown("**The Bug Busters** © 2025 | IT3051 Fundamentals of Data Mining |")
