import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib


def train_and_save_model(data_path, model_path):
    print("Loading processed data...")
    df = pd.read_csv(data_path)

    # Drop any ID columns
    df.drop(columns=[c for c in ['MLS', 'MLSå¨ #', 'id']
            if c in df.columns], inplace=True, errors='ignore')

    feature_columns = [
        'property-beds', 'property-baths', 'Square Footage', 'Acreage',
        'latitude', 'longitude', 'has_fireplace', 'has_garage',
        'dist_to_toronto_km', 'addressRegion', 'Property Type',
        'Basement', 'Fireplace', 'Heating', 'Parking', 'region_median_price'
    ]

    X = df[feature_columns].copy()
    numeric_cols = X.select_dtypes(include='number').columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, df['price'], test_size=0.2, random_state=42)

    # Log transform for tree-based models (BEST PERFORMANCE)
    y_train_log = np.log1p(y_train_raw)
    y_test_log = np.log1p(y_test_raw)

    models = {
        # Uses raw price
        'LinearRegression': (LinearRegression(), y_train_raw, y_test_raw),
        'DecisionTree': (DecisionTreeRegressor(random_state=42, max_depth=20), y_train_log, y_test_log),
        'RandomForest': (RandomForestRegressor(random_state=42, n_jobs=-1), y_train_log, y_test_log),
        'SVR': (SVR(C=1000, gamma='scale'), y_train_log, y_test_log),
        'KNN': (KNeighborsRegressor(n_neighbors=7), y_train_log, y_test_log),
        'XGBoost': (XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=9,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
                    y_train_log, y_test_log)
    }

    best_r2 = -999
    best_model = None
    results = {}

    for name, (model, y_train, y_test) in models.items():
        print(f"Training {name}...", end="")

        try:
            if name == 'RandomForest':
                grid = GridSearchCV(model, {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 20],
                    'min_samples_split': [2, 5]
                }, cv=3, scoring='r2', n_jobs=-1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                print(f"\nBest RF params: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
                print()

            pred = model.predict(X_test)
            if name != 'LinearRegression':
                pred = np.expm1(pred)  # Reverse log
                actual = np.expm1(y_test)
            else:
                actual = y_test

            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))

            print(f"{name} → R²: {r2:.4f} | MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f}")
            results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}

            if r2 > best_r2:
                best_r2 = r2
                best_model = model

        except Exception as e:
            print(f"\n{name} failed: {e}")

    joblib.dump(best_model, model_path)
    print(
        f"\nBEST MODEL: {best_model.__class__.__name__} → R² = {best_r2:.4f}")
    print(f"Model saved → {model_path}")

    # FINAL GOLD TABLE
    print("\n" + "="*95)
    print("           THE BUG BUSTERS - FINAL RESULTS ")
    print("="*95)
    print(f"{'Model':<18} {'R²':<10} {'MAE':<18} {'RMSE'}")
    print("-"*95)
    for name in ['LinearRegression', 'DecisionTree', 'RandomForest', 'SVR', 'KNN', 'XGBoost']:
        if name in results:
            r = results[name]
            star = " ← BEST" if r['R2'] == max(
                [v['R2'] for v in results.values()]) else ""
            print(
                f"{name:<18} {r['R2']:.4f}     ${r['MAE']:,.0f}        ${r['RMSE']:,.0f}{star}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')
    model_path = os.path.join(current_dir, 'model.pkl')
    train_and_save_model(data_path, model_path)
