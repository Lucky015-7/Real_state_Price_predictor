import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# -----------------------------------
# Logging configuration
# -----------------------------------
logging.basicConfig(
    level=logging.INFO,  # Show INFO, WARNING, ERROR messages
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_and_prepare_data(file_path):
    """Load raw data, clean it, handle missing values, engineer features, 
    and prepare dataset for ML model training"""

    # Load dataset (skip bad lines to prevent crashes)
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, quotechar='"',
                     escapechar='\\', on_bad_lines='skip')

    # Replace "Unknown" string with NaN for consistency
    df.replace('Unknown', np.nan, inplace=True)

    # Convert numeric-like columns to numbers

    for col in ['property-sqft', 'Square Footage', 'price',
                'property-beds', 'property-baths', 'Acreage']:
        if col in df.columns:
            # Remove commas, extract numbers from strings
            df[col] = df[col].astype(str).str.replace(
                ',', '').str.extract(r'(\d+\.?\d*)')
            # Convert to numeric (invalid values → NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

            if df[col].isna().any():
                logging.warning(f"NaN values found in {col} after conversion")

    # Handle missing values
    numerical_cols = ['price', 'property-beds', 'property-baths',
                      'property-sqft', 'Square Footage', 'Acreage',
                      'latitude', 'longitude']

    categorical_cols = ['addressRegion', 'Property Type', 'Basement',
                        'Fireplace', 'Heating', 'Parking', 'Exterior',
                        'Exterior Features', 'Features',
                        'Fireplace Features', 'Flooring',
                        'Parking Features', 'Roof', 'Sewer',
                        'Subdivision', 'Type']

    # Fill NaNs in numerical columns with median
    for col in numerical_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f"Filled {col} NaNs with median {median_val}")

    # Fill NaNs in categorical columns with mode
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(
            )[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            logging.info(f"Filled {col} NaNs with mode {mode_val}")

    # Remove duplicates

    initial_rows = len(df)
    df.drop_duplicates(
        subset=['latitude', 'longitude', 'price', 'Square Footage'], inplace=True
    )
    logging.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Handle price outliers

    if 'price' in df.columns:
        lower, upper = df['price'].quantile([0.01, 0.99])
        df['price'] = df['price'].clip(lower, upper)
        logging.info(f"Clipped price outliers to range [{lower}, {upper}]")

    # Feature Engineering
    ##
    # Price per square foot
    if 'price' in df.columns and 'Square Footage' in df.columns:
        df['price_per_sqft'] = df.apply(
            lambda row: row['price'] / row['Square Footage']
            if pd.notna(row['price']) and pd.notna(row['Square Footage']) and row['Square Footage'] > 0
            else np.nan,
            axis=1
        )

    # Fill NaNs in price_per_sqft with median
    if df['price_per_sqft'].isna().any():
        median_pps = df['price_per_sqft'].median()
        df['price_per_sqft'] = df['price_per_sqft'].fillna(median_pps)
        logging.info(f"Filled price_per_sqft NaNs with median {median_pps}")
    else:
        logging.warning(
            "Could not compute price_per_sqft due to missing columns")

    # Binary feature: has fireplace
    if 'Fireplace' in df.columns:
        df['has_fireplace'] = df['Fireplace'].apply(
            lambda x: 1 if pd.notna(x) and x != 'No' else 0
        )

    # Binary feature: has garage
    if 'Parking' in df.columns:
        df['has_garage'] = df['Parking'].str.contains(
            'Garage', na=False).astype(int
                                       )

    # Distance to Toronto (Haversine formula)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        def haversine(lat1, lon1, lat2, lon2):
            """Calculate distance between two points (in km) using Haversine formula"""
            R = 6371  # Earth radius in km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
                np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        toronto_lat, toronto_lon = 43.7, -79.4  # Toronto coordinates

        df['dist_to_toronto_km'] = df.apply(
            lambda row: haversine(row['latitude'], row['longitude'],
                                  toronto_lat, toronto_lon)
            if pd.notna(row['latitude']) and pd.notna(row['longitude'])
            else np.nan,
            axis=1
        )

        # Fill NaNs with median distance
        if df['dist_to_toronto_km'].isna().any():
            median_dist = df['dist_to_toronto_km'].median()
            df['dist_to_toronto_km'] = df['dist_to_toronto_km'].fillna(
                median_dist)
            logging.info(
                f"Filled dist_to_toronto_km NaNs with median {median_dist}")
    else:
        logging.warning(
            "Latitude/Longitude not found. Cannot calculate distance to Toronto.")
        df['dist_to_toronto_km'] = 100.0  # Default value if missing

    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = le.fit_transform(df[col])

    # Drop irrelevant columns

    columns_to_drop = ['priceCurrency', 'MLSå¨ #', 'description',
                       'streetAddress', 'addressLocality', 'postalCode', 'Property Tax']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Final check for missing values

    if df.isnull().sum().sum() > 0:
        logging.warning(f"Remaining NaNs found: {df.isnull().sum()}")

    return df


# Script entry point
if __name__ == "__main__":
    import os
    # Paths for input (raw) and output (processed) data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, '..', 'data', 'cleaned_data.csv')
    output_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')

    # Run preprocessing
    data = load_and_prepare_data(input_path)

    # Save processed data
    data.to_csv(output_path, index=False)

    # Print summary for validation
    print("Data preprocessing complete. First 5 rows:")
    print(data.head())
    print("Missing values per column:\n", data.isnull().sum())
