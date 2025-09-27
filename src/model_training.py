import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib


def train_and_save_model(data_path, model_path):
    """Train multiple models and save the best one with comprehensive evaluation"""
    try:
        # Load data with validation
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate required columns
        required_columns = ['property-beds', 'property-baths', 'Square Footage', 'Acreage', 
                           'latitude', 'longitude', 'price_per_sqft', 'has_fireplace', 
                           'has_garage', 'dist_to_toronto_km', 'addressRegion', 'Property Type', 
                           'Basement', 'Fireplace', 'Heating', 'Parking', 'price']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Features (expanded)
        feature_columns = required_columns[:-1]  # All except 'price'
        X = df[feature_columns]
        y = df['price']
        
        print(f"Training with {len(feature_columns)} features")
        print(f"Target variable (price) range: ${y.min():,.2f} - ${y.max():,.2f}")

        # Clean features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Check for remaining NaN values
        if X.isnull().sum().sum() > 0:
            print(f"Warning: {X.isnull().sum().sum()} NaN values remaining after cleaning")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Multiple models with improved parameters
        models = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=20),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100)
        }

        best_model = None
        best_r2 = -np.inf
        model_results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                if name == 'RandomForest':  # Enhanced hyperparameter tuning
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='r2')
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    print(f"Best parameters: {grid.best_params_}")
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                model_results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'model': model
                }
                
                print(f"{name} - MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}, R2: {r2:.4f}")

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        if best_model is None:
            raise ValueError("No models were successfully trained")
        
        # Save the best model
        joblib.dump(best_model, model_path)
        
        # Print comprehensive results
        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Best model: {best_model.__class__.__name__} with R2 = {best_r2:.4f}")
        print(f"Model saved to: {model_path}")
        
        # Print all model results
        print(f"\nAll Model Results:")
        for name, results in model_results.items():
            print(f"{name:15} - R2: {results['R2']:.4f}, MAE: ${results['MAE']:,.2f}, RMSE: ${results['RMSE']:,.2f}")
        
        return best_model, model_results
        
    except Exception as e:
        print(f"Error in model training: {e}")
        raise


if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')
    model_path = os.path.join(current_dir, 'model.pkl')
    train_and_save_model(data_path, model_path)
