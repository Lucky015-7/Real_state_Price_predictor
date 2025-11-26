# Real Estate Price Predictor

A comprehensive machine learning application for predicting real estate prices using various algorithms and interactive visualizations.

### Price Prediction

- **Interactive Web Interface**: User-friendly Streamlit app for property price prediction
- **Multiple Input Parameters**: Bedrooms, bathrooms, square footage, location, and additional features
- **Real-time Validation**: Input validation with helpful error messages
- **Comprehensive Analysis**: Price per square foot, market comparison, and percentile ranking

### Advanced Visualizations

- **Interactive Charts**: Plotly-powered visualizations with zoom, hover, and filtering
- **Correlation Analysis**: Interactive heatmaps showing feature relationships
- **Regional Analysis**: Price distribution and comparison across different regions
- **Property Features Impact**: Scatter plots showing how features affect pricing
- **Interactive Maps**: Geographic visualization of property locations and prices
- **Model Performance**: Feature importance and model evaluation metrics

## Final Model Performance

| Model             | R² Score   | MAE          | RMSE         | Notes                  |
| ----------------- | ---------- | ------------ | ------------ | ---------------------- |
| XGBoost           | **0.7320** | **$176,452** | **$359,076** | **BEST & FINAL MODEL** |
| Random Forest     | 0.7131     | $183,005     | $371,500     |                        |
| Decision Tree     | 0.5338     | $242,795     | $473,543     |                        |
| Linear Regression | 0.3982     | $332,252     | $538,056     | Baseline               |

### Machine Learning Models

- **Multiple Algorithms**: Linear Regression, Decision Tree, and Random Forest, XGBoost
- **Hyperparameter Tuning**: Grid search optimization for best performance
- **Comprehensive Evaluation**: MAE, RMSE, and R² metrics for model comparison
- **Feature Engineering**: Distance calculations,region_median_price and categorical encoding

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Real_state_Price_predictor
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**

   - Place your cleaned data in `data/cleaned_data.csv`
   - Run data preprocessing: `python src/data_preprocessing.py`
   - Train the model: `python src/model_training.py`

5. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

## 📁 Project Structure

```
Real_state_Price_predictor/
├── data/
│   ├── cleaned_data.csv          # Raw data
│   └── processed_data.csv         # Preprocessed data
├── src/
│   ├── app.py                    # Main Streamlit application
│   ├── data_preprocessing.py     # Data cleaning and preparation
│   ├── model_training.py         # Model training and evaluation
│   ├── visualization.py          # Interactive visualizations
│   └── model.pkl                 # Trained model
├── visualizations/               # Saved visualization images
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Key Improvements Made

### 1. **Enhanced Visualizations**

- ✅ Interactive Plotly charts with zoom and hover
- ✅ Comprehensive correlation heatmaps
- ✅ Regional price analysis with filtering
- ✅ Interactive property location maps
- ✅ Feature importance visualizations
- ✅ Model performance metrics

### 2. **Better Code Organization**

- ✅ Modular function structure
- ✅ Proper error handling and validation
- ✅ Relative file paths for portability
- ✅ Comprehensive documentation
- ✅ Input validation and user feedback

### 3. **Improved User Experience**

- ✅ Better UI layout with columns and sections
- ✅ Helpful tooltips and guidance
- ✅ Loading spinners and progress indicators
- ✅ Comprehensive price analysis
- ✅ Market context and percentile ranking

### 4. **Enhanced Model Training**

- ✅ Better hyperparameter tuning
- ✅ Comprehensive model evaluation
- ✅ Error handling and validation
- ✅ Detailed training logs
- ✅ Multiple algorithm comparison

## 📊 Data Requirements

The application expects the following columns in your dataset:

### Required Columns

- `property-beds`: Number of bedrooms
- `property-baths`: Number of bathrooms
- `Square Footage`: Property size in square feet
- `Acreage`: Lot size in acres
- `latitude`, `longitude`: Geographic coordinates
- `price`: Property price (target variable)
- `addressRegion`: Province/region
- `Property Type`: Type of property

### Optional Columns

- `Basement`, `Fireplace`, `Heating`, `Parking`: Property features

## 🎨 Visualization Features

### Interactive Charts

- **Correlation Heatmap**: Shows relationships between numeric features
- **Price Distribution**: Histogram and box plots with statistics
- **Regional Analysis**: Price comparison across regions
- **Feature Impact**: Scatter plots showing feature-price relationships
- **Interactive Maps**: Geographic visualization with property details
- **Model Analysis**: Feature importance and performance metrics

### Customization Options

- Region filtering for focused analysis
- Multiple chart types for different insights
- Downloadable static versions
- Responsive design for different screen sizes

## 🔍 Model Performance

The application trains multiple models and selects the best performer:

- **Linear Regression**: Baseline model for comparison
- **Decision Tree**: Non-linear relationships
- **Random Forest**: Ensemble method with feature importance

### Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination

## Customization

### Adding New Features

1. Update the feature columns in `model_training.py`
2. Modify the input form in `app.py`
3. Update the visualization functions in `visualization.py`

### Modifying Visualizations

- All visualization functions are modular and can be customized
- Plotly charts support extensive customization options
- Easy to add new chart types or modify existing ones

## Future Enhancements

- [ ] Additional machine learning models (XGBoost, Neural Networks)
- [ ] Real-time data integration
- [ ] Advanced feature engineering
- [ ] Model explainability (SHAP values)
- [ ] Automated model retraining
- [ ] Export functionality for predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues:

1. Check the error messages in the application
2. Verify your data format matches the requirements
3. Ensure all dependencies are installed
4. Check the console output for detailed error information
