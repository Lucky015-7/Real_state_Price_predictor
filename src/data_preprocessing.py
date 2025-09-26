import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(file_path):
    # Load data with error handling for quotes and commas
    df = pd.read_csv(file_path, quotechar='"',
                     escapechar='\\', on_bad_lines='skip')

    # Replace 'Unknown' with NaN
    df.replace('Unknown', np.nan, inplace=True)

    # Convert to numeric, clean strings with better error handling
    for col in ['property-sqft', 'Square Footage', 'price', 'property-beds', 'property-baths', 'Acreage']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                ',', '').str.extract(r'(\d+\.?\d*)')
            # Convert to float, NaN for invalid
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values without inplace warning
    numerical_cols = ['price', 'property-beds', 'property-baths',
                      'property-sqft', 'Square Footage', 'Acreage', 'latitude', 'longitude']
    categorical_cols = ['addressRegion', 'Property Type',
                        'Basement', 'Fireplace', 'Heating', 'Parking']

    for col in numerical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle outliers (clip price to 1-99 percentile)
    if 'price' in df.columns:
        lower, upper = df['price'].quantile([0.01, 0.99])
        df['price'] = df['price'].clip(lower, upper)

    # Feature Engineering
    if 'price' in df.columns and 'Square Footage' in df.columns and not df['Square Footage'].isna().all():
        df['price_per_sqft'] = df['price'] / df['Square Footage']

    if 'Fireplace' in df.columns:
        df['has_fireplace'] = df['Fireplace'].apply(
            lambda x: 1 if pd.notna(x) and x != 'No' else 0)

    if 'Parking' in df.columns:
        df['has_garage'] = df['Parking'].str.contains(
            'Garage', na=False).astype(int)

    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            # Ensure no NaN before encoding
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = le.fit_transform(df[col])

    # Drop irrelevant columns
    df.drop(columns=['MLS', 'description', 'streetAddress',
            'addressLocality', 'postalCode'], errors='ignore', inplace=True)

    return df


# Usage
file_path = r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\cleaned_data.csv"
data = load_and_prepare_data(file_path)
data.to_csv(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\processed_data.csv", index=False)
print("Data preprocessing complete. First 5 rows:")
print(data.head())
print("Missing values per column:\n", data.isnull().sum())
