# Advanced Features Documentation

<!--
Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
-->

## Overview

This document describes all the advanced features added to the Energy Consumption Dataset project.

## 1. Machine Learning Forecasting (`forecasting.py`)

### Features
- **Linear Regression Model**: Fast and interpretable forecasting
- **Random Forest Model**: Ensemble method for improved accuracy
- **Feature Engineering**: 
  - Time-based features (hour, day, month, cyclical encoding)
  - Lag features (1 hour, 24 hours, 168 hours)
  - Rolling statistics (mean, std)
- **Future Forecasting**: Predict consumption for next 24 hours
- **Model Comparison**: Side-by-side performance metrics

### Usage
```bash
python forecasting.py
```

### Output
- Model performance metrics (MAE, RMSE, R², MAPE)
- Model comparison table
- 24-hour forecast saved to `forecast_24h.csv`

## 2. Anomaly Detection (`anomaly_detection.py`)

### Detection Methods
1. **IQR (Interquartile Range)**: Statistical outlier detection
2. **Z-Score**: Standard deviation based detection
3. **Isolation Forest**: Machine learning based detection
4. **Time Series**: Rolling window based detection

### Features
- Comprehensive detection combining multiple methods
- Anomaly scoring and classification
- Summary statistics by household and hour
- Export detected anomalies

### Usage
```bash
python anomaly_detection.py
```

### Output
- `energy_consumption_with_anomalies.csv`: Full dataset with anomaly flags
- `anomalies_only.csv`: Only anomalous records
- Summary statistics and breakdowns

## 3. Advanced Time Series Analysis (`advanced_analysis.py`)

### Analysis Components
1. **Time Series Decomposition**:
   - Trend component
   - Seasonal component
   - Residual component

2. **Autocorrelation Analysis**:
   - Calculate autocorrelation function
   - Identify patterns and dependencies
   - Up to 48 lags

3. **Trend Detection**:
   - Mann-Kendall test
   - Trend direction (increasing/decreasing)
   - Statistical significance

4. **Stationarity Testing**:
   - Variance ratio test
   - Mean change analysis
   - Stationary/non-stationary classification

5. **Seasonality Strength**:
   - Hourly seasonality
   - Daily seasonality
   - Monthly seasonality

### Usage
```bash
python advanced_analysis.py
```

### Output
- Comprehensive analysis results
- Trend and stationarity metrics
- Seasonality strength indicators

## 4. Data Preprocessing (`preprocessing.py`)

### Preprocessing Steps
1. **Missing Value Handling**:
   - Forward fill
   - Backward fill
   - Mean/Median imputation
   - Drop missing values

2. **Outlier Removal**:
   - IQR method
   - Z-score method

3. **Data Normalization**:
   - Min-max scaling
   - Standard scaling

4. **Feature Engineering**:
   - Time features (year, month, day, quarter, etc.)
   - Cyclical encoding (sin/cos transformations)
   - Lag features
   - Rolling window features
   - Weekend/month start/end flags

### Usage
```bash
python preprocessing.py
```

### Output
- `energy_consumption_processed.csv`: Preprocessed dataset
- Preprocessing summary and statistics

## 5. Model Evaluation (`model_evaluation.py`)

### Evaluation Metrics
- **MAE** (Mean Absolute Error): Average absolute difference
- **MSE** (Mean Squared Error): Penalizes larger errors
- **RMSE** (Root Mean Squared Error): Standard deviation of residuals
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **MBE** (Mean Bias Error): Average prediction bias
- **CV(RMSE)**: Coefficient of variation of RMSE

### Features
- Model comparison tables
- Prediction vs actual plots
- Residual analysis plots
- Comprehensive evaluation reports

### Usage
```bash
python model_evaluation.py
```

### Output
- Model comparison charts
- Prediction plots
- Evaluation reports (`model_evaluation_report.txt`)

## Enhanced HTML Demo (`index.html`)

### New Features
- **Additional Charts**:
  - Seasonal patterns (monthly)
  - Household comparison (doughnut chart)
  
- **Advanced Features Section**:
  - Machine Learning showcase
  - Anomaly Detection showcase
  - Time Series Analysis showcase

- **Interactive Visualizations**:
  - Real-time data loading
  - Dynamic statistics
  - Multiple chart types

## Complete Workflow

### 1. Generate Data
```bash
python generate_data.py
```

### 2. Basic Analysis
```bash
python analysis.py
python visualization.py
```

### 3. Advanced Analysis
```bash
python preprocessing.py
python advanced_analysis.py
python anomaly_detection.py
python forecasting.py
python model_evaluation.py
```

### 4. View Results
- Open `index.html` in browser for interactive demo
- Check generated CSV files for processed data
- Review generated plots and reports

## Dependencies

All advanced features require:
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0

Install with:
```bash
pip install -r requirements.txt
```

## Contact

For questions or support:
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

