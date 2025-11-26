# src/visualization.py  (or wherever you keep it)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ADD THIS FOR WIDE LAYOUT & TITLE
st.set_page_config(
    page_title="The Bug Busters - Visualizations", layout="wide")
st.title("The Bug Busters")
st.markdown("### Interactive Real Estate Data Explorer")


def load_data_and_model():
    """Load data and model with proper error handling"""
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
        st.stop()
    except Exception as e:
        st.error(f"Error loading data/model: {e}")
        st.stop()


def create_correlation_heatmap(df):
    """Create an interactive correlation heatmap"""
    st.subheader("Correlation Heatmap (Numeric Features)")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation heatmap.")
        return

    corr_matrix = numeric_df.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )

    fig.update_layout(width=800, height=600, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show static version for download"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax,
                    square=True, fmt='.2f')
        plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)


def create_price_distribution(df):
    """Create comprehensive price distribution analysis"""
    st.subheader("Price Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x='price', nbins=50,
            title="Price Distribution",
            labels={'price': 'Price (CAD)', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, y='price',
            title="Price Distribution (Box Plot)",
            labels={'price': 'Price (CAD)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Statistics")
    price_stats = df['price'].describe()
    st.dataframe(price_stats.round(2))


def create_regional_analysis(df):
    """Create regional price analysis"""
    st.subheader("Regional Price Analysis")

    regions = df['addressRegion'].unique()
    selected_regions = st.multiselect(
        "Select regions to compare:",
        regions,
        default=regions[:3] if len(regions) >= 3 else regions
    )

    if selected_regions:
        filtered_df = df[df['addressRegion'].isin(selected_regions)]

        fig = px.box(
            filtered_df,
            x='addressRegion',
            y='price',
            title="Price Distribution by Region",
            labels={'addressRegion': 'Region', 'price': 'Price (CAD)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        avg_prices = filtered_df.groupby('addressRegion')['price'].agg(
            ['mean', 'median', 'count']).round(2)
        avg_prices.columns = ['Average Price',
                              'Median Price', 'Number of Properties']
        st.subheader("Regional Price Summary")
        st.dataframe(avg_prices)


def create_property_features_analysis(df):
    """Analyze property features impact on price"""
    st.subheader("Property Features Impact on Price")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df, x='property-beds', y='price',
            title="Price vs Number of Bedrooms",
            labels={'property-beds': 'Number of Bedrooms',
                    'price': 'Price (CAD)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df, x='Square Footage', y='price',
            title="Price vs Square Footage",
            labels={'Square Footage': 'Square Footage', 'price': 'Price (CAD)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def create_model_performance_visualization(model, df):
    """Create model performance and feature importance visualization"""
    st.subheader("Model Performance & Feature Importance")

    try:
        if hasattr(model, 'feature_importances_'):
            # EXACT FINAL FEATURES FROM YOUR TRAINING — NO LEAKAGE!
            feature_columns = [
                'property-beds', 'property-baths', 'Square Footage', 'Acreage',
                'latitude', 'longitude', 'has_fireplace', 'has_garage',
                'dist_to_toronto_km', 'addressRegion', 'Property Type',
                'Basement', 'Fireplace', 'Heating', 'Parking', 'region_median_price'
            ]

            importances = model.feature_importances_

            if len(importances) != len(feature_columns):
                st.error(
                    f"Feature count mismatch! Model has {len(importances)} features, but list has {len(feature_columns)}")
                return

            fi_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)

            fig = px.bar(
                fi_df.tail(10),
                x="Importance",
                y="Feature",
                orientation='h',
                title="Top 10 Feature Importance",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=550, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                "**Top Features:** `region_median_price`, `Square Footage`, `dist_to_toronto_km`, `latitude` — location & size dominate pricing!")

        else:
            st.info(
                "Feature importance only available for tree-based models (XGBoost/RF).")

    except Exception as e:
        st.error(f"Error in feature importance: {e}")


def create_interactive_map(df):
    """Create an interactive map of properties"""
    st.subheader("Property Location Map")

    sample_size = min(1500, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    fig = px.scatter_mapbox(
        sample_df,
        lat="latitude",
        lon="longitude",
        color="price",
        size="Square Footage",
        hover_data=["property-beds", "property-baths", "addressRegion"],
        color_continuous_scale="Plasma",
        mapbox_style="open-street-map",
        title=f"Property Prices Across Canada ({sample_size:,} properties)",
        zoom=4.5
    )

    fig.update_layout(height=650, margin={"r": 0, "t": 50, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main visualization function"""
    st.header("Interactive Real Estate Visualizations")

    model, df = load_data_and_model()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Regional", "Features", "Model", "Map", "Correlation"
    ])

    with tab1:
        create_price_distribution(df)

    with tab2:
        create_regional_analysis(df)

    with tab3:
        create_property_features_analysis(df)

    with tab4:
        create_model_performance_visualization(model, df)

    with tab5:
        create_interactive_map(df)

    with tab6:
        create_correlation_heatmap(df)


def show_visualizations():
    """Function to be called from the main Streamlit app"""
    main()
