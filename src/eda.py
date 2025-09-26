import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_preprocessing import load_and_prepare_data

df = pd.read_csv(r"E:\FDMproject\data\processed_data.csv")  # Use processed

# Correlation (Mining)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(r"E:\FDMproject\visualizations\correlation.png")

# Clustering (e.g., by price/region)
features = ['price', 'dist_to_toronto_km']  # Adjust
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(df[features])
sns.scatterplot(x='dist_to_toronto_km', y='price', hue='cluster', data=df)
plt.title('Clusters by Price and Distance')
plt.savefig(r"E:\FDMproject\visualizations\clusters.png")

# PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(
    df.select_dtypes(include=[np.number]).dropna())
df_pca = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
sns.scatterplot(x='PC1', y='PC2', data=df_pca)
plt.title('PCA of Features')
plt.savefig(r"E:\FDMproject\visualizations\pca.png")

# Price by Region (Viz)
if 'addressRegion' in df.columns:
    sns.boxplot(x='addressRegion', y='price', data=df)
    plt.title('Price by Region')
    plt.savefig(r"E:\FDMproject\visualizations\price_by_region.png")

print("EDA complete. Visuals saved in visualizations folder.")
