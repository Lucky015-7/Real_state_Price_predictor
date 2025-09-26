import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_preprocessing import load_and_prepare_data

# Load and preprocess data
df = load_and_prepare_data(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\cleaned_data.csv")

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\correlation.png")
plt.close()  # Close figure to free memory

# Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(numeric_df['price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.savefig(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\price_dist.png")
plt.close()

# Clustering by Price and Square Footage
features = ['price', 'Square Footage']
if all(col in numeric_df.columns for col in features):
    kmeans = KMeans(n_clusters=3, random_state=42)
    numeric_df['cluster'] = kmeans.fit_predict(numeric_df[features])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Square Footage', y='price',
                    hue='cluster', data=numeric_df, palette='deep')
    plt.title('Price vs Square Footage Clusters')
    plt.savefig(
        r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\clusters.png")
    plt.close()
else:
    print("Warning: Required features for clustering not found in numeric data.")

# PCA for dimensionality reduction
if len(numeric_df.columns) > 1:
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(numeric_df)
    df_pca = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=df_pca)
    plt.title('PCA of Features')
    plt.savefig(
        r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\pca.png")
    plt.close()
else:
    print("Warning: Insufficient numeric features for PCA.")

print("EDA and visualizations saved.")
