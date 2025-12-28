"""
Energy Consumption Dataset - Anomaly Detection

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Advanced anomaly detection for energy consumption patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Anomaly detection class for energy consumption data.
    """
    
    def __init__(self, df):
        """
        Initialize anomaly detector with data.
        
        Args:
            df: pandas.DataFrame containing energy consumption data
        """
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.anomalies = None
        self.scaler = StandardScaler()
    
    def detect_statistical_outliers(self, method='iqr', threshold=3):
        """
        Detect outliers using statistical methods.
        
        Args:
            method: 'iqr' for Interquartile Range or 'zscore' for Z-score
            threshold: Threshold for z-score method
        
        Returns:
            pandas.DataFrame: Data with anomaly flags
        """
        df_result = self.df.copy()
        
        if method == 'iqr':
            Q1 = df_result['consumption_kwh'].quantile(0.25)
            Q3 = df_result['consumption_kwh'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_result['is_anomaly_iqr'] = (
                (df_result['consumption_kwh'] < lower_bound) | 
                (df_result['consumption_kwh'] > upper_bound)
            )
            
        elif method == 'zscore':
            mean = df_result['consumption_kwh'].mean()
            std = df_result['consumption_kwh'].std()
            z_scores = np.abs((df_result['consumption_kwh'] - mean) / std)
            df_result['is_anomaly_zscore'] = z_scores > threshold
        
        return df_result
    
    def detect_isolation_forest(self, contamination=0.1, random_state=42):
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
        
        Returns:
            pandas.DataFrame: Data with anomaly flags
        """
        df_result = self.df.copy()
        
        # Prepare features
        features = ['consumption_kwh', 'hour', 'day_of_week', 'temperature']
        features = [f for f in features if f in df_result.columns]
        
        X = df_result[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        df_result['is_anomaly_iso'] = iso_forest.fit_predict(X_scaled) == -1
        df_result['anomaly_score'] = iso_forest.score_samples(X_scaled)
        
        return df_result
    
    def detect_time_series_anomalies(self, window=24, threshold=2):
        """
        Detect anomalies based on time series patterns.
        
        Args:
            window: Rolling window size for moving average
            threshold: Number of standard deviations for threshold
        
        Returns:
            pandas.DataFrame: Data with anomaly flags
        """
        df_result = self.df.copy()
        df_result = df_result.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate rolling statistics
        df_result['rolling_mean'] = df_result.groupby('household_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df_result['rolling_std'] = df_result.groupby('household_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Detect anomalies
        df_result['upper_bound'] = df_result['rolling_mean'] + threshold * df_result['rolling_std']
        df_result['lower_bound'] = df_result['rolling_mean'] - threshold * df_result['rolling_std']
        
        df_result['is_anomaly_ts'] = (
            (df_result['consumption_kwh'] > df_result['upper_bound']) |
            (df_result['consumption_kwh'] < df_result['lower_bound'])
        )
        
        return df_result
    
    def comprehensive_detection(self):
        """
        Run all detection methods and combine results.
        
        Returns:
            pandas.DataFrame: Data with comprehensive anomaly flags
        """
        print("Running comprehensive anomaly detection...")
        
        # Run all methods
        df_iqr = self.detect_statistical_outliers(method='iqr')
        df_zscore = self.detect_statistical_outliers(method='zscore', threshold=3)
        df_iso = self.detect_isolation_forest(contamination=0.1)
        df_ts = self.detect_time_series_anomalies(window=24, threshold=2)
        
        # Combine results
        df_result = self.df.copy()
        df_result['is_anomaly_iqr'] = df_iqr['is_anomaly_iqr'] if 'is_anomaly_iqr' in df_iqr.columns else False
        df_result['is_anomaly_zscore'] = df_zscore['is_anomaly_zscore'] if 'is_anomaly_zscore' in df_zscore.columns else False
        df_result['is_anomaly_iso'] = df_iso['is_anomaly_iso'] if 'is_anomaly_iso' in df_iso.columns else False
        df_result['is_anomaly_ts'] = df_ts['is_anomaly_ts'] if 'is_anomaly_ts' in df_ts.columns else False
        df_result['anomaly_score'] = df_iso['anomaly_score'] if 'anomaly_score' in df_iso.columns else 0
        
        # Combined flag (anomaly if detected by at least 2 methods)
        anomaly_cols = ['is_anomaly_iqr', 'is_anomaly_zscore', 'is_anomaly_iso', 'is_anomaly_ts']
        df_result['is_anomaly'] = df_result[anomaly_cols].sum(axis=1) >= 2
        
        self.anomalies = df_result[df_result['is_anomaly']]
        
        return df_result
    
    def get_anomaly_summary(self):
        """
        Get summary statistics of detected anomalies.
        
        Returns:
            dict: Summary statistics
        """
        if self.anomalies is None:
            return None
        
        summary = {
            'total_anomalies': len(self.anomalies),
            'anomaly_percentage': (len(self.anomalies) / len(self.df)) * 100,
            'avg_anomaly_consumption': self.anomalies['consumption_kwh'].mean(),
            'avg_normal_consumption': self.df[~self.df['is_anomaly']]['consumption_kwh'].mean() if 'is_anomaly' in self.df.columns else self.df['consumption_kwh'].mean(),
            'anomalies_by_household': self.anomalies.groupby('household_id').size().to_dict(),
            'anomalies_by_hour': self.anomalies.groupby('hour').size().to_dict()
        }
        
        return summary

def main():
    """
    Main function to demonstrate anomaly detection.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - ANOMALY DETECTION")
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
    
    # Initialize detector
    detector = AnomalyDetector(df)
    
    # Run comprehensive detection
    df_with_anomalies = detector.comprehensive_detection()
    
    # Get summary
    summary = detector.get_anomaly_summary()
    
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 60)
    print(f"\nTotal Anomalies Detected: {summary['total_anomalies']:,}")
    print(f"Anomaly Percentage: {summary['anomaly_percentage']:.2f}%")
    print(f"\nAverage Consumption:")
    print(f"  Normal: {summary['avg_normal_consumption']:.2f} kWh")
    print(f"  Anomalies: {summary['avg_anomaly_consumption']:.2f} kWh")
    
    print("\nAnomalies by Household:")
    for household, count in summary['anomalies_by_household'].items():
        print(f"  {household}: {count:,} anomalies")
    
    # Save results
    df_with_anomalies.to_csv('energy_consumption_with_anomalies.csv', index=False)
    print("\nResults saved to energy_consumption_with_anomalies.csv")
    
    # Save only anomalies
    detector.anomalies.to_csv('anomalies_only.csv', index=False)
    print("Anomalies saved to anomalies_only.csv")
    
    print("\n" + "=" * 60)
    print("Anomaly detection complete!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

