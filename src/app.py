import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from eda import *  # If you make eda functions importable

model = joblib.load(r"E:\FDMproject\model.pkl")

st.title("🏠 Canadian Property Price Predictor")

page = st.sidebar.selectbox(
    "Choose Page", ["Predict Price", "Upload & Analyze Data", "Visualizations"])

if page == "Predict Price":
    st.header("Property Details Questionnaire")
    beds = st.number_input("How many bedrooms?", 1, 10, 3)
    baths = st.number_input("How many bathrooms?", 1, 10, 2)
    sqft = st.number_input("Square footage?", 500, 10000, 1500)
    acreage = st.number_input("Acreage?", 0.0, 10.0, 0.5)
    lat = st.number_input("Latitude?", 40.0, 70.0, 45.0)
    lon = st.number_input("Longitude?", -140.0, -50.0, -79.0)
    region = st.selectbox("Province/Region?",
                          df['addressRegion'].unique())  # From data
    property_type = st.selectbox(
        "Property Type?", df['Property Type'].unique())
    # Add more: basement, fireplace, etc.

    if st.button("Get Price Suggestion"):
        input_df = pd.DataFrame({
            'property-beds': [beds], 'property-baths': [baths], 'Square Footage': [sqft],
            'Acreage': [acreage], 'latitude': [lat], 'longitude': [lon],
            # Add engineered: price_per_sqft (placeholder 0), has_fireplace (0/1 checkbox), etc.
            # Encode if needed
            'addressRegion': [region], 'Property Type': [property_type]
        })
        # Preprocess input to match training (e.g., encode, add dist_to_toronto)
        # ... (call a function to align)
        pred = model.predict(input_df)[0]
        st.success(f"Estimated Price: ${pred:,.2f} CAD")

elif page == "Upload & Analyze Data":
    st.header("Upload Dataset")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df_upload = pd.read_csv(file)
        st.dataframe(df_upload.head())
        if st.button("Run Analysis (EDA)"):
            # Run EDA code here or call functions
            st.write("Summary:", df_upload.describe())
            # Add clustering/PCA results

elif page == "Visualizations":
    st.header("Interactive Visuals")
    # Load plots or generate on-the-fly
    st.image(r"E:\FDMproject\visualizations\correlation.png")
    region_filter = st.selectbox(
        "Filter by Region", df['addressRegion'].unique())
    filtered = df[df['addressRegion'] == region_filter]
    fig, ax = plt.subplots()
    sns.boxplot(x='property-beds', y='price', data=filtered, ax=ax)
    st.pyplot(fig)
    # Add more interactive (e.g., price dist, feature importance)
