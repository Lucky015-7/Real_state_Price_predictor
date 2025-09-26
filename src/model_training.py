import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from data_preprocessing import load_and_prepare_data

# Load data (use absolute or corrected relative path)
df = load_and_prepare_data(r"E:\FDMproject\data\cleaned_data.csv")

# Check if data loaded (add this to handle None)
if df is None:
    print("Failed to load data. Exiting.")
    exit()

# Select features (adjust column names based on your data)
# From the head() output, columns might be "property-beds" or similar—verify!
feature_columns = ["property-beds", "Square Footage", "latitude", "longitude"]
if all(col in df.columns for col in feature_columns):
    X = df[feature_columns]
else:
    print("Missing columns. Available columns:", df.columns.tolist())
    print(df.head())  # Debug print
    exit()

y = df["price"]

# Handle NaN values (important for ML)
df = df.dropna(subset=feature_columns + ["price"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
