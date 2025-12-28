# Energy Consumption Dataset

<!--
Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
-->

Smart meter energy consumption dataset with hourly electricity usage patterns for demand forecasting and energy analytics.

## Description

This dataset contains hourly energy consumption data from smart meters with seasonal patterns, peak hours, and consumption trends. Perfect for energy demand forecasting, load prediction, and smart grid applications.

## Features

- Hourly consumption data
- Seasonal patterns
- Peak hour identification
- Multiple households
- Time series ready format

## Technologies

- CSV
- JSON
- Pandas
- Time Series Analysis

## Difficulty Level

Intermediate

## Dataset Structure

The dataset includes:
- `energy_consumption.csv` - Main dataset in CSV format
- `energy_consumption.json` - Dataset in JSON format
- `index.html` - Interactive demo page with visualizations

### Python Scripts

**Basic Analysis:**
- `generate_data.py` - Generate synthetic energy consumption data
- `analysis.py` - Basic statistical analysis and insights
- `visualization.py` - Create charts and visualizations

**Advanced Features:**
- `forecasting.py` - Machine learning forecasting models (Linear Regression, Random Forest)
- `anomaly_detection.py` - Multiple anomaly detection methods (IQR, Z-score, Isolation Forest, Time Series)
- `advanced_analysis.py` - Advanced time series analysis (decomposition, autocorrelation, trend detection)
- `preprocessing.py` - Data preprocessing and feature engineering utilities
- `model_evaluation.py` - Comprehensive model evaluation metrics and comparison

## Usage

### Basic Analysis

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('energy_consumption.csv')

# Basic statistics
print(df.describe())

# Run analysis script
python analysis.py

# Generate visualizations
python visualization.py
```

### Advanced Features

**Machine Learning Forecasting:**
```python
python forecasting.py
```
- Linear Regression model
- Random Forest model
- 24-hour future forecasting
- Model comparison metrics

**Anomaly Detection:**
```python
python anomaly_detection.py
```
- IQR method
- Z-score method
- Isolation Forest
- Time series anomaly detection

**Advanced Time Series Analysis:**
```python
python advanced_analysis.py
```
- Time series decomposition
- Autocorrelation analysis
- Trend detection (Mann-Kendall test)
- Stationarity testing
- Seasonality strength calculation

**Data Preprocessing:**
```python
python preprocessing.py
```
- Missing value handling
- Outlier removal
- Data normalization
- Feature engineering
- Lag and rolling features

**Model Evaluation:**
```python
python model_evaluation.py
```
- Comprehensive metrics (MAE, RMSE, R², MAPE, MBE)
- Model comparison
- Prediction plots
- Evaluation reports

### Data Format

- **timestamp**: Date and time of measurement
- **household_id**: Unique identifier for each household
- **consumption_kwh**: Energy consumption in kilowatt-hours
- **temperature**: Outdoor temperature (for correlation analysis)
- **hour**: Hour of the day (0-23)
- **day_of_week**: Day of the week (0-6, Monday=0)

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Advanced Features

### 1. Machine Learning Forecasting
- **Linear Regression**: Fast and interpretable forecasting model
- **Random Forest**: Ensemble method for better accuracy
- **Feature Engineering**: Time-based features, lag features, rolling statistics
- **Future Forecasting**: Predict consumption for next 24 hours

### 2. Anomaly Detection
- **Statistical Methods**: IQR and Z-score based detection
- **Isolation Forest**: Machine learning based anomaly detection
- **Time Series Methods**: Rolling window based anomaly detection
- **Comprehensive Detection**: Combines multiple methods for robust detection

### 3. Advanced Time Series Analysis
- **Decomposition**: Trend, seasonal, and residual components
- **Autocorrelation**: Identify patterns and dependencies
- **Trend Detection**: Mann-Kendall test for trend significance
- **Stationarity Testing**: Variance ratio and mean change analysis
- **Seasonality Analysis**: Strength of hourly, daily, and monthly patterns

### 4. Data Preprocessing
- **Missing Value Handling**: Forward fill, backward fill, mean, median, or drop
- **Outlier Removal**: IQR or Z-score based methods
- **Normalization**: Min-max or standard scaling
- **Feature Engineering**: Time features, cyclical encoding, lag features, rolling statistics

### 5. Model Evaluation
- **Comprehensive Metrics**: MAE, RMSE, R², MAPE, MBE, CV(RMSE)
- **Visual Comparisons**: Prediction plots, residual plots, model comparison charts
- **Evaluation Reports**: Detailed text reports with recommendations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Free to use for educational and research purposes.

## Contact

For questions or support:
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

