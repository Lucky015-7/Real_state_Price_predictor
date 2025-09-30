import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_preprocessing import load_and_prepare_data

# Step 1: Load and preprocess data

# We call our custom preprocessing function to clean the raw dataset
df = load_and_prepare_data(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\cleaned_data.csv"
)

# Select only numeric columns for correlation/analysis
numeric_df = df.select_dtypes(include=['int64', 'float64'])


# Step 2: Correlation Heatmap

# Shows relationships between numeric features (e.g., price, sqft, beds, baths)
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\correlation.png"
)
plt.close()  # Close figure to free memory


# Step 3: Price Distribution

# Shows how property prices are distributed (normal, skewed, etc.)
plt.figure(figsize=(10, 6))
sns.histplot(numeric_df['price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.savefig(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\price_dist.png"
)
plt.close()


# Step 4: Clustering (K-Means) by Price & Square Footage

# Groups properties into clusters to identify market segments (low, medium, high value)
features = ['price', 'Square Footage']
if all(col in numeric_df.columns for col in features):
    kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters
    numeric_df['cluster'] = kmeans.fit_predict(numeric_df[features])

    # Scatter plot with cluster colors
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Square Footage', y='price',
                    hue='cluster', data=numeric_df, palette='deep')
    plt.title('Price vs Square Footage Clusters')
    plt.savefig(
        r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\clusters.png"
    )
    plt.close()
else:
    print("Warning: Required features for clustering not found in numeric data.")


# Step 5: PCA (Dimensionality Reduction)

# Reduces all numeric features into 2 principal components
# Useful for visualizing high-dimensional data
if len(numeric_df.columns) > 1:
    # Clean up infinities/NaNs before PCA
    numeric_df_clean = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()

    if numeric_df_clean.shape[0] > 0 and numeric_df_clean.shape[1] > 1:
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(numeric_df_clean)

        # Create a DataFrame with the 2 PCA components
        df_pca = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])

        # Scatter plot of PCA results
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', data=df_pca)
        plt.title('PCA of Features')
        plt.savefig(
            r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\pca.png"
        )
        plt.close()
    else:
        print("Warning: Not enough clean numeric data for PCA.")
else:
    print("Warning: Insufficient numeric features for PCA.")
