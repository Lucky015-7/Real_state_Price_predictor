import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import numpy as np


def train_and_save_model(data_path, model_path):
    df = pd.read_csv(data_path)

    # Features (expanded)
    feature_columns = ['property-beds', 'property-baths', 'Square Footage', 'Acreage', 'latitude', 'longitude',
                       'price_per_sqft', 'has_fireplace', 'has_garage', 'dist_to_toronto_km',
                       'addressRegion', 'Property Type', 'Basement', 'Fireplace', 'Heating', 'Parking']  # Encoded

    X = df[feature_columns]
    y = df['price']

    # Clean features
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Multiple models
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(random_state=42)
    }

    best_model = None
    best_r2 = -np.inf  # allow negative R²

    for name, model in models.items():
        if name == 'RandomForest':  # Tune
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    joblib.dump(best_model, model_path)
    print(
        f"Best model: {best_model.__class__.__name__} with R2 = {best_r2:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    data_path = r"E:\FDMproject\data\processed_data.csv"
    model_path = r"E:\FDMproject\model.pkl"
    train_and_save_model(data_path, model_path)
