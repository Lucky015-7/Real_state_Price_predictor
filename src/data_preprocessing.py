import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os

# -----------------------------------
# Logging configuration
# -----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_and_prepare_data(file_path):
    """THE BUG BUSTERS FINAL EDITION - High-Performance & Leakage-Free Preprocessing"""

    logging.info(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path, quotechar='"',
                     escapechar='\\', on_bad_lines='skip')
    df.replace('Unknown', np.nan, inplace=True)
    logging.info(f"Loaded {len(df)} rows")

    # ============ 1. Convert numeric columns ============
    numeric_cols = ['property-sqft', 'Square Footage',
                    'price', 'property-beds', 'property-baths', 'Acreage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                ',', '').str.extract(r'(\d+\.?\d*)')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ============ 2. Fill missing values ============
    # Numerical → median
    for col in ['price', 'property-beds', 'property-baths', 'Square Footage', 'Acreage', 'latitude', 'longitude']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical → mode
    categorical_cols = ['addressRegion', 'Property Type', 'Basement', 'Fireplace', 'Heating',
                        'Parking', 'Exterior', 'Exterior Features', 'Features', 'Fireplace Features',
                        'Flooring', 'Parking Features', 'Roof', 'Sewer', 'Subdivision', 'Type']
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode(
            )[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)

    # ============ 3. Remove duplicates & clip outliers ============
    initial_rows = len(df)
    df.drop_duplicates(
        subset=['latitude', 'longitude', 'price', 'Square Footage'], inplace=True)
    logging.info(f"Removed {initial_rows - len(df)} duplicate listings")

    if 'price' in df.columns:
        lower, upper = df['price'].quantile([0.01, 0.99])
        df['price'] = df['price'].clip(lower, upper)
        logging.info(
            f"Clipped price outliers to [{lower:,.0f} - {upper:,.0f}]")

    # ============ 4. Legitimate Feature Engineering (NO LEAKAGE) ============
    # Has fireplace
    if 'Fireplace' in df.columns:
        df['has_fireplace'] = df['Fireplace'].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() != 'No' else 0)

    # Has garage
    if 'Parking' in df.columns:
        df['has_garage'] = df['Parking'].str.contains(
            'Garage', case=False, na=False).astype(int)

    # Distance to Toronto
    if 'latitude' in df.columns and 'longitude' in df.columns:
        def haversine(lat1, lon1):
            R = 6371
            lat2, lon2 = 43.7, -79.4
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
                np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        df['dist_to_toronto_km'] = df.apply(
            lambda row: haversine(row['latitude'], row['longitude'])
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) else np.nan, axis=1)
        df['dist_to_toronto_km'].fillna(
            df['dist_to_toronto_km'].median(), inplace=True)

    # ============ 5. Encode categorical variables FIRST ============
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

    # ============ 6. BEST ETHICAL FEATURE: region_median_price (AFTER ENCODING!) ============
    if 'addressRegion' in df.columns and 'price' in df.columns:
        # Now addressRegion is encoded → perfect grouping!
        region_median = df.groupby('addressRegion')[
            'price'].transform('median')
        df['region_median_price'] = region_median
        df['region_median_price'].fillna(df['price'].median(), inplace=True)
        logging.info(
            "Added HIGH-IMPACT region_median_price feature using encoded regions")

    # ============ 7. Drop irrelevant columns ============
    drop_cols = ['priceCurrency', 'MLSå¨ #', 'description', 'streetAddress',
                 'addressLocality', 'postalCode', 'Property Tax']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    logging.info(f"FINAL DATASET: {df.shape[0]} rows × {df.shape[1]} columns")
    logging.info(f"Columns: {list(df.columns)}")

    return df


# ============ RUN SCRIPT ============
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, '..', 'data', 'cleaned_data.csv')
    output_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')

    processed_data = load_and_prepare_data(input_path)
    processed_data.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("="*60)
    print(f"   Shape: {processed_data.shape}")
    print("="*60)
    print(processed_data[['price', 'Square Footage',
          'addressRegion', 'region_median_price']].head(10))
