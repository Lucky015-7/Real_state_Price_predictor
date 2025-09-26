import pandas as pd


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Convert Square Footage to numeric
    if "Square Footage" in df.columns:
        df["Square Footage"] = df["Square Footage"].str.extract(r'(\d+)')
        df["Square Footage"] = pd.to_numeric(
            df["Square Footage"], errors="coerce")

    # Create price_per_sqft
    if "price" in df.columns and "Square Footage" in df.columns:
        df["price_per_sqft"] = df["price"] / df["Square Footage"]

    # Drop useless column if it exists
    df = df.drop(columns=["MLS"], errors="ignore")

    return df


# Specify the file path (use raw string to handle backslashes)
file_path = r"E:\FDMproject\data\cleaned_data.csv"

# Load and prepare the data
data = load_and_prepare_data(file_path)

# If data loaded successfully, proceed with any initial checks
if data is not None:
    print("First 5 rows of the processed dataset:")
    print(data.head())
