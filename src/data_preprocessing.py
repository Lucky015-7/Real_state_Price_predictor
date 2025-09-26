import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(file_path):
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, quotechar='"',
                     escapechar='\\', on_bad_lines='skip')

    df.replace('Unknown', np.nan, inplace=True)

    for col in ['property-sqft', 'Square Footage', 'price', 'property-beds', 'property-baths', 'Acreage']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                ',', '').str.extract(r'(\d+\.?\d*)')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                logging.warning(f"NaN values found in {col} after conversion")

    numerical_cols = ['price', 'property-beds', 'property-baths',
                      'property-sqft', 'Square Footage', 'Acreage', 'latitude', 'longitude']
    categorical_cols = ['addressRegion', 'Property Type', 'Basement', 'Fireplace', 'Heating', 'Parking', 'Exterior', 'Exterior Features',
                        'Features', 'Fireplace Features', 'Flooring', 'Parking Features', 'Roof', 'Sewer', 'Subdivision', 'Type']

    for col in numerical_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f"Filled {col} NaNs with median {median_val}")

    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(
            )[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            logging.info(f"Filled {col} NaNs with mode {mode_val}")

    initial_rows = len(df)
    df.drop_duplicates(
        subset=['latitude', 'longitude', 'price', 'Square Footage'], inplace=True)
    logging.info(f"Removed {initial_rows - len(df)} duplicate rows")

    if 'price' in df.columns:
        lower, upper = df['price'].quantile([0.01, 0.99])
        df['price'] = df['price'].clip(lower, upper)
        logging.info(f"Clipped price outliers to range [{lower}, {upper}]")

    if 'price' in df.columns and 'Square Footage' in df.columns:
        df['price_per_sqft'] = df.apply(
            lambda row: row['price'] / row['Square Footage']
            if pd.notna(row['price']) and pd.notna(row['Square Footage']) and row['Square Footage'] > 0
            else np.nan,
            axis=1
        )

    # Fill remaining NaNs in price_per_sqft with median
    if df['price_per_sqft'].isna().any():
        median_pps = df['price_per_sqft'].median()
        df['price_per_sqft'] = df['price_per_sqft'].fillna(median_pps)
        logging.info(f"Filled price_per_sqft NaNs with median {median_pps}")
    else:
        logging.warning(
            "Could not compute price_per_sqft due to missing columns")

    if 'Fireplace' in df.columns:
        df['has_fireplace'] = df['Fireplace'].apply(
            lambda x: 1 if pd.notna(x) and x != 'No' else 0)

    if 'Parking' in df.columns:
        df['has_garage'] = df['Parking'].str.contains(
            'Garage', na=False).astype(int)

    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = le.fit_transform(df[col])

    # Drop columns with minimal data or irrelevance, matching exact name
    columns_to_drop = ['priceCurrency', 'MLSå¨ #', 'description',
                       'streetAddress', 'addressLocality', 'postalCode', 'Property Tax']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Final check for NaNs
    if df.isnull().sum().sum() > 0:
        logging.warning(f"Remaining NaNs found: {df.isnull().sum()}")

    return df


file_path = r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\cleaned_data.csv"
data = load_and_prepare_data(file_path)
data.to_csv(r"C:\Users\lakhi\OneDrive\Desktop\IRWAproj\Real_state_Price_predictor\data\processed_data.csv", index=False)
print("Data preprocessing complete. First 5 rows:")
print(data.head())
print("Missing values per column:\n", data.isnull().sum())
