import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Handle 'Unknown' as NaN
    df.replace('Unknown', np.nan, inplace=True)

    # Convert to numeric, clean strings
    for col in ['property-sqft', 'Square Footage', 'price', 'property-beds', 'property-baths', 'Acreage']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                ',', '').str.extract(r'(\d+\.?\d*)').astype(float)

    # Fill missing values
    numerical_cols = ['price', 'property-beds', 'property-baths',
                      'property-sqft', 'Square Footage', 'Acreage', 'latitude', 'longitude']
    categorical_cols = ['addressRegion', 'Property Type',
                        'Basement', 'Fireplace', 'Heating', 'Parking']

    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Handle duplicates
    df.drop_duplicates(inplace=True)

    # Outliers: Clip price to 1-99 percentile
    if 'price' in df.columns:
        lower, upper = df['price'].quantile([0.01, 0.99])
        df['price'] = np.clip(df['price'], lower, upper)

    # Feature Engineering
    if 'price' in df.columns and 'Square Footage' in df.columns:
        df['price_per_sqft'] = df['price'] / df['Square Footage']

    if 'Fireplace' in df.columns:
        df['has_fireplace'] = df['Fireplace'].apply(
            lambda x: 1 if pd.notna(x) and x != 'No' else 0)

    if 'Parking' in df.columns:
        df['has_garage'] = df['Parking'].str.contains(
            'Garage', na=False).astype(int)

    # Proximity to amenities (demo: distance to Toronto center ~43.65, -79.38)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        def haversine(lat1, lon1, lat2=43.65, lon2=-79.38):
            R = 6371  # Earth radius km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
                np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        df['dist_to_toronto_km'] = haversine(df['latitude'], df['longitude'])

    # Encode categoricals
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Drop useless columns
    df.drop(columns=['MLS', 'description', 'streetAddress'],
            errors='ignore', inplace=True)

    return df


# Usage
file_path = r"E:\FDMproject\data\cleaned_data.csv"
data = load_and_prepare_data(file_path)
data.to_csv(r"E:\FDMproject\data\processed_data.csv",
            index=False)  # Save processed
print(data.head())
