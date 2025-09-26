import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import numpy as np

df = pd.read_csv(r"E:\FDMproject\data\processed_data.csv")

# Features (expanded)
feature_columns = ['property-beds', 'property-baths', 'Square Footage', 'Acreage', 'latitude', 'longitude',
                   'price_per_sqft', 'has_fireplace', 'has_garage', 'dist_to_toronto_km',
                   'addressRegion', 'Property Type', 'Basement', 'Fireplace', 'Heating', 'Parking']  # Encoded

X = df[feature_columns]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Multiple models
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = 0

for name, model in models.items():
    if name == 'RandomForest':  # Tune
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

joblib.dump(best_model, r"E:\FDMproject\model.pkl")
print("Best model saved.")
