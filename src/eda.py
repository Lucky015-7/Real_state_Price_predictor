# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# LOAD DATA
df = pd.read_csv(
    r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\cleaned_data.csv")

print(f"Original rows: {len(df)}")

# FORCE CONVERT TO NUMERIC (CRITICAL!)
cols_to_numeric = ['price', 'Square Footage', 'property-beds',
                   'property-baths', 'Acreage', 'latitude', 'longitude']
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(
            r'[^0-9.-]', '', regex=True), errors='coerce')

# DROP ONLY ROWS THAT ARE COMPLETELY USELESS
df = df.dropna(subset=['price', 'Square Footage'])  # Only need these two
print(f"After dropping rows with missing price/sqft: {len(df)} rows")

# FILL REMAINING MISSING VALUES WITH MEDIAN (instead of dropna)
for col in ['property-beds', 'property-baths', 'Acreage', 'latitude', 'longitude']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# FINAL FILTER (realistic properties)
df = df[df['price'].between(100_000, 8_000_000)]
df = df[df['Square Footage'].between(600, 15_000)]

print(f"Final clean dataset for EDA: {len(df)} properties")

# ENSURE WE HAVE DATA
if len(df) == 0:
    print("ERROR: No data left after cleaning. Check your cleaned_data.csv")
    exit()

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = ['price', 'Square Footage', 'property-beds',
             'property-baths', 'Acreage', 'latitude', 'longitude']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap of Property Features', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\correlation.png",
            dpi=300, bbox_inches='tight')
plt.close()

# 2. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True, color='#3498DB')
plt.title('Distribution of Property Prices', fontsize=14)
plt.xlabel('Price (CAD)')
plt.tight_layout()
plt.savefig(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\price_dist.png",
            dpi=300, bbox_inches='tight')
plt.close()

# 3. Price vs Square Footage
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Square Footage',
                y='price', alpha=0.6, color='#27AE60')
plt.title('Price vs Square Footage', fontsize=14)
plt.xlabel('Square Footage')
plt.ylabel('Price (CAD)')
plt.tight_layout()
plt.savefig(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\price_vs_sqft.png",
            dpi=300, bbox_inches='tight')
plt.close()

# 4. K-Means Clustering (UNSUPERVISED)
features = ['Square Footage', 'property-beds',
            'property-baths', 'latitude', 'longitude']
X = df[features]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(11, 7))
sns.scatterplot(data=df, x='Square Footage', y='price',
                hue='cluster', palette='deep', alpha=0.7, s=70)
plt.title('Market Segments via K-Means Clustering', fontsize=14)
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\clusters.png",
            dpi=300, bbox_inches='tight')
plt.close()

# 5. PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                hue=df['cluster'], palette='deep', alpha=0.8)
plt.title(
    f'PCA Visualization (Explained: {pca.explained_variance_ratio_.sum():.1%})', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\visualizations\pca.png",
            dpi=300, bbox_inches='tight')
plt.close()
