"""
Energy Consumption Dataset - Data Preprocessing Utilities

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Data preprocessing and cleaning utilities for energy consumption data.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessor:
    """
    Data preprocessing class for energy consumption data.
    """
    
    def __init__(self, df):
        """
        Initialize preprocessor with data.
        
        Args:
            df: pandas.DataFrame containing energy consumption data
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.preprocessing_steps = []
    
    def handle_missing_values(self, method='forward_fill'):
        """
        Handle missing values in the dataset.
        
        Args:
            method: 'forward_fill', 'backward_fill', 'mean', 'median', or 'drop'
        
        Returns:
            self: For method chaining
        """
        missing_before = self.df.isnull().sum().sum()
        
        if method == 'forward_fill':
            self.df = self.df.ffill()
        elif method == 'backward_fill':
            self.df = self.df.bfill()
        elif method == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif method == 'median':
            self.df = self.df.fillna(self.df.median())
        elif method == 'drop':
            self.df = self.df.dropna()
        
        missing_after = self.df.isnull().sum().sum()
        self.preprocessing_steps.append(f"Handled missing values ({method}): {missing_before} -> {missing_after}")
        
        return self
    
    def remove_outliers(self, method='iqr', columns=None):
        """
        Remove outliers from the dataset.
        
        Args:
            method: 'iqr' for Interquartile Range or 'zscore' for Z-score
            columns: List of columns to process (None for all numeric columns)
        
        Returns:
            self: For method chaining
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        rows_before = len(self.df)
        
        if method == 'iqr':
            for col in columns:
                if col in self.df.columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                if col in self.df.columns:
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    self.df = self.df[z_scores < 3]
        
        rows_after = len(self.df)
        self.preprocessing_steps.append(f"Removed outliers ({method}): {rows_before} -> {rows_after} rows")
        
        return self
    
    def normalize_data(self, method='min_max', columns=None):
        """
        Normalize data to a specific range.
        
        Args:
            method: 'min_max' or 'standard'
            columns: List of columns to normalize (None for all numeric columns)
        
        Returns:
            self: For method chaining
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude timestamp-related columns
            columns = [col for col in columns if col not in ['hour', 'day_of_week', 'month', 'day', 'year']]
        
        if method == 'min_max':
            for col in columns:
                if col in self.df.columns:
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:
                        self.df[f'{col}_normalized'] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            for col in columns:
                if col in self.df.columns:
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        self.df[f'{col}_standardized'] = (self.df[col] - mean_val) / std_val
        
        self.preprocessing_steps.append(f"Normalized data ({method})")
        
        return self
    
    def create_time_features(self):
        """
        Create additional time-based features.
        
        Returns:
            self: For method chaining
        """
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Extract time features
            self.df['year'] = self.df['timestamp'].dt.year
            self.df['month'] = self.df['timestamp'].dt.month
            self.df['day'] = self.df['timestamp'].dt.day
            self.df['day_of_year'] = self.df['timestamp'].dt.dayofyear
            iso_cal = self.df['timestamp'].dt.isocalendar()
            self.df['week_of_year'] = iso_cal['week'] if isinstance(iso_cal, pd.DataFrame) else iso_cal.week
            self.df['quarter'] = self.df['timestamp'].dt.quarter
            self.df['is_weekend'] = (self.df['timestamp'].dt.dayofweek >= 5).astype(int)
            self.df['is_month_start'] = self.df['timestamp'].dt.is_month_start.astype(int)
            self.df['is_month_end'] = self.df['timestamp'].dt.is_month_end.astype(int)
            
            # Cyclical encoding
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
            self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        self.preprocessing_steps.append("Created time features")
        
        return self
    
    def create_lag_features(self, lags=[1, 24, 168]):
        """
        Create lag features for time series.
        
        Args:
            lags: List of lag periods to create
        
        Returns:
            self: For method chaining
        """
        for lag in lags:
            self.df[f'consumption_lag_{lag}'] = self.df.groupby('household_id')['consumption_kwh'].shift(lag)
        
        self.preprocessing_steps.append(f"Created lag features: {lags}")
        
        return self
    
    def create_rolling_features(self, windows=[24, 168]):
        """
        Create rolling window features.
        
        Args:
            windows: List of window sizes
        
        Returns:
            self: For method chaining
        """
        for window in windows:
            self.df[f'consumption_rolling_mean_{window}'] = self.df.groupby('household_id')['consumption_kwh'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.df[f'consumption_rolling_std_{window}'] = self.df.groupby('household_id')['consumption_kwh'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        self.preprocessing_steps.append(f"Created rolling features: {windows}")
        
        return self
    
    def get_summary(self):
        """
        Get preprocessing summary.
        
        Returns:
            dict: Summary of preprocessing steps
        """
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_added': self.df.shape[1] - self.original_shape[1],
            'preprocessing_steps': self.preprocessing_steps
        }
    
    def get_data(self):
        """
        Get processed dataframe.
        
        Returns:
            pandas.DataFrame: Processed data
        """
        return self.df

def main():
    """
    Main function to demonstrate preprocessing.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - DATA PREPROCESSING")
    print("=" * 60)
    print("Project: Energy Consumption Dataset")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("=" * 60 + "\n")
    
    # Load data
    try:
        df = pd.read_csv('energy_consumption.csv')
        print(f"Loaded {len(df):,} records")
        print(f"Original shape: {df.shape}")
    except FileNotFoundError:
        print("Error: energy_consumption.csv not found. Please generate data first.")
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df)
    
    # Apply preprocessing steps
    print("\nApplying preprocessing steps...")
    preprocessor.handle_missing_values(method='forward_fill')
    preprocessor.create_time_features()
    preprocessor.create_lag_features(lags=[1, 24, 168])
    preprocessor.create_rolling_features(windows=[24, 168])
    
    # Get summary
    summary = preprocessor.get_summary()
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original shape: {summary['original_shape']}")
    print(f"Final shape: {summary['final_shape']}")
    print(f"Rows removed: {summary['rows_removed']}")
    print(f"Columns added: {summary['columns_added']}")
    
    print("\nPreprocessing steps:")
    for step in summary['preprocessing_steps']:
        print(f"  ✓ {step}")
    
    # Save processed data
    processed_df = preprocessor.get_data()
    processed_df.to_csv('energy_consumption_processed.csv', index=False)
    print("\nProcessed data saved to energy_consumption_processed.csv")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

