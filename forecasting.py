"""
Energy Consumption Dataset - Forecasting Models

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Advanced forecasting models for energy consumption prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnergyForecaster:
    """
    Advanced forecasting class for energy consumption prediction.
    """
    
    def __init__(self, df):
        """
        Initialize forecaster with data.
        
        Args:
            df: pandas.DataFrame containing energy consumption data
        """
        self.df = df.copy()
        self.models = {}
        self.scaler = StandardScaler()
        self.prepare_features()
    
    def prepare_features(self):
        """
        Create time-based features for forecasting.
        """
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Time features
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['day_of_year'] = self.df['timestamp'].dt.dayofyear
        iso_cal = self.df['timestamp'].dt.isocalendar()
        self.df['week_of_year'] = iso_cal['week'] if isinstance(iso_cal, pd.DataFrame) else iso_cal.week
        
        # Cyclical encoding for time features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Lag features
        for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
            self.df[f'consumption_lag_{lag}'] = self.df.groupby('household_id')['consumption_kwh'].shift(lag)
        
        # Rolling statistics
        self.df['consumption_rolling_mean_24'] = self.df.groupby('household_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        self.df['consumption_rolling_std_24'] = self.df.groupby('household_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )
        
        # Drop rows with NaN from lag features
        self.df = self.df.dropna().reset_index(drop=True)
    
    def train_test_split(self, test_size=0.2):
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(self.df) * (1 - test_size))
        
        feature_cols = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'temperature', 'consumption_lag_1', 'consumption_lag_24', 'consumption_lag_168',
            'consumption_rolling_mean_24', 'consumption_rolling_std_24'
        ]
        
        # Filter available columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[feature_cols]
        y = self.df['consumption_kwh']
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self):
        """
        Train Linear Regression model.
        
        Returns:
            dict: Model performance metrics
        """
        X_train, X_test, y_train, y_test = self.train_test_split()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'model_name': 'Linear Regression',
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.models['linear_regression'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test.values
        }
        
        return metrics
    
    def train_random_forest(self, n_estimators=100):
        """
        Train Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
        
        Returns:
            dict: Model performance metrics
        """
        X_train, X_test, y_train, y_test = self.train_test_split()
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': 'Random Forest',
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.models['random_forest'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test.values
        }
        
        return metrics
    
    def forecast_future(self, model_name='random_forest', periods=24):
        """
        Forecast future consumption.
        
        Args:
            model_name: Name of the model to use
            periods: Number of hours to forecast
        
        Returns:
            pandas.DataFrame: Forecasted values
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        model = self.models[model_name]['model']
        last_row = self.df.iloc[-1].copy()
        
        forecasts = []
        current_data = last_row.copy()
        
        for i in range(periods):
            # Prepare features for prediction
            feature_cols = [
                'hour', 'day_of_week', 'month', 'day_of_year',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'temperature', 'consumption_lag_1', 'consumption_lag_24', 'consumption_lag_168',
                'consumption_rolling_mean_24', 'consumption_rolling_std_24'
            ]
            feature_cols = [col for col in feature_cols if col in self.df.columns]
            
            X_pred = current_data[feature_cols].values.reshape(1, -1)
            
            # Scale if using linear regression
            if model_name == 'linear_regression':
                X_pred = self.scaler.transform(X_pred)
            
            pred = model.predict(X_pred)[0]
            forecasts.append(pred)
            
            # Update for next iteration
            current_data['consumption_lag_1'] = pred
            current_data['hour'] = (current_data['hour'] + 1) % 24
            if current_data['hour'] == 0:
                current_data['day_of_week'] = (current_data['day_of_week'] + 1) % 7
        
        # Create forecast dataframe
        last_timestamp = pd.to_datetime(self.df['timestamp'].iloc[-1])
        forecast_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=periods, freq='H')
        
        forecast_df = pd.DataFrame({
            'timestamp': forecast_dates,
            'forecasted_consumption': forecasts
        })
        
        return forecast_df
    
    def compare_models(self):
        """
        Compare all trained models.
        
        Returns:
            pandas.DataFrame: Comparison of model metrics
        """
        if not self.models:
            print("No models trained yet. Train models first.")
            return None
        
        comparison = []
        for model_name, model_data in self.models.items():
            comparison.append(model_data['metrics'])
        
        return pd.DataFrame(comparison)

def main():
    """
    Main function to demonstrate forecasting capabilities.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - FORECASTING MODELS")
    print("=" * 60)
    print("Project: Energy Consumption Dataset")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("=" * 60 + "\n")
    
    # Load data
    try:
        df = pd.read_csv('energy_consumption.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df):,} records")
    except FileNotFoundError:
        print("Error: energy_consumption.csv not found. Please generate data first.")
        return
    
    # Initialize forecaster
    print("\nInitializing forecaster and preparing features...")
    forecaster = EnergyForecaster(df)
    
    # Train models
    print("\nTraining Linear Regression model...")
    lr_metrics = forecaster.train_linear_regression()
    print(f"  MAE: {lr_metrics['mae']:.3f} kWh")
    print(f"  RMSE: {lr_metrics['rmse']:.3f} kWh")
    print(f"  R² Score: {lr_metrics['r2']:.3f}")
    print(f"  MAPE: {lr_metrics['mape']:.2f}%")
    
    print("\nTraining Random Forest model...")
    rf_metrics = forecaster.train_random_forest(n_estimators=100)
    print(f"  MAE: {rf_metrics['mae']:.3f} kWh")
    print(f"  RMSE: {rf_metrics['rmse']:.3f} kWh")
    print(f"  R² Score: {rf_metrics['r2']:.3f}")
    print(f"  MAPE: {rf_metrics['mape']:.2f}%")
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison = forecaster.compare_models()
    print(comparison.to_string(index=False))
    
    # Generate forecast
    print("\n" + "=" * 60)
    print("GENERATING 24-HOUR FORECAST")
    print("=" * 60)
    forecast = forecaster.forecast_future(model_name='random_forest', periods=24)
    print("\nNext 24 Hours Forecast:")
    print(forecast.head(10).to_string(index=False))
    print(f"\n... and {len(forecast) - 10} more hours")
    
    # Save forecast
    forecast.to_csv('forecast_24h.csv', index=False)
    print("\nForecast saved to forecast_24h.csv")
    
    print("\n" + "=" * 60)
    print("Forecasting complete!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

