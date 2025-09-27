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

def load_data_and_model():
    """Load data and model with proper error handling"""
    try:
        # Use relative paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.pkl')
        data_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')
        
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
    st.subheader("🔗 Correlation Heatmap (Numeric Features)")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation heatmap.")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create interactive plotly heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show seaborn version for download
    if st.checkbox("Show static version for download"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, 
                   square=True, fmt='.2f')
        plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

def create_price_distribution(df):
    """Create comprehensive price distribution analysis"""
    st.subheader("💰 Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with KDE
        fig = px.histogram(
            df, x='price', nbins=50,
            title="Price Distribution",
            labels={'price': 'Price (CAD)', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df, y='price',
            title="Price Distribution (Box Plot)",
            labels={'price': 'Price (CAD)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price statistics
    st.subheader("📊 Price Statistics")
    price_stats = df['price'].describe()
    st.dataframe(price_stats.round(2))

def create_regional_analysis(df):
    """Create regional price analysis"""
    st.subheader("🏘️ Regional Price Analysis")
    
    # Region selector
    regions = df['addressRegion'].unique()
    selected_regions = st.multiselect(
        "Select regions to compare:", 
        regions, 
        default=regions[:3] if len(regions) >= 3 else regions
    )
    
    if selected_regions:
        filtered_df = df[df['addressRegion'].isin(selected_regions)]
        
        # Box plot by region
        fig = px.box(
            filtered_df, 
            x='addressRegion', 
            y='price',
            title="Price Distribution by Region",
            labels={'addressRegion': 'Region', 'price': 'Price (CAD)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average price by region
        avg_prices = filtered_df.groupby('addressRegion')['price'].agg(['mean', 'median', 'count']).round(2)
        avg_prices.columns = ['Average Price', 'Median Price', 'Number of Properties']
        st.subheader("📈 Regional Price Summary")
        st.dataframe(avg_prices)

def create_property_features_analysis(df):
    """Analyze property features impact on price"""
    st.subheader("🏠 Property Features Impact on Price")
    
    # Bedrooms vs Price
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df, x='property-beds', y='price',
            title="Price vs Number of Bedrooms",
            labels={'property-beds': 'Number of Bedrooms', 'price': 'Price (CAD)'},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df, x='Square Footage', y='price',
            title="Price vs Square Footage",
            labels={'Square Footage': 'Square Footage', 'price': 'Price (CAD)'},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature impact analysis
    st.subheader("🔍 Feature Impact Analysis")
    
    # Create feature impact visualization
    features_to_analyze = ['property-beds', 'property-baths', 'Square Footage', 'Acreage']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Price vs {feature.replace("-", " ").title()}' for feature in features_to_analyze]
    )
    
    for i, feature in enumerate(features_to_analyze):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Scatter(
                x=df[feature], 
                y=df['price'],
                mode='markers',
                name=feature,
                opacity=0.6
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def create_model_performance_visualization(model, df):
    """Create model performance and feature importance visualization"""
    st.subheader("🤖 Model Performance & Feature Importance")
    
    try:
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_columns = ['property-beds', 'property-baths', 'Square Footage', 'Acreage', 
                             'latitude', 'longitude', 'price_per_sqft', 'has_fireplace', 
                             'has_garage', 'dist_to_toronto_km', 'addressRegion', 'Property Type', 
                             'Basement', 'Fireplace', 'Heating', 'Parking']
            
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                fi_df, 
                x="Importance", 
                y="Feature",
                orientation='h',
                title="Feature Importance (Random Forest)",
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features
            st.subheader("🏆 Top 5 Most Important Features")
            top_features = fi_df.tail(5)
            st.dataframe(top_features)
            
        else:
            st.warning("Feature importances are only available for tree-based models.")
            
    except Exception as e:
        st.error(f"Error creating model visualization: {e}")

def create_interactive_map(df):
    """Create an interactive map of properties"""
    st.subheader("🗺️ Property Location Map")
    
    # Sample data for performance (if dataset is large)
    if len(df) > 1000:
        sample_df = df.sample(n=1000, random_state=42)
        st.info(f"Showing 1000 random properties from {len(df)} total properties")
    else:
        sample_df = df
    
    # Create map
    fig = px.scatter_mapbox(
        sample_df,
        lat="latitude",
        lon="longitude",
        color="price",
        size="Square Footage",
        hover_data=["property-beds", "property-baths", "addressRegion"],
        color_continuous_scale="Viridis",
        mapbox_style="open-street-map",
        title="Property Locations and Prices",
        labels={'price': 'Price (CAD)', 'Square Footage': 'Square Footage'}
    )
    
    fig.update_layout(
        height=600,
        mapbox=dict(
            center=dict(lat=sample_df['latitude'].mean(), lon=sample_df['longitude'].mean()),
            zoom=6
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main visualization function"""
    st.header("📊 Interactive Real Estate Visualizations")
    
    # Load data and model
    model, df = load_data_and_model()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🏘️ Regional Analysis", "🏠 Property Features", 
        "🤖 Model Analysis", "🗺️ Location Map", "📊 Correlation"
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

# This function should be called from the main app
def show_visualizations():
    """Function to be called from the main Streamlit app"""
    main()
