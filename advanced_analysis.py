"""
Energy Consumption Dataset - Advanced Time Series Analysis

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Advanced time series analysis including decomposition, autocorrelation, and trend analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class AdvancedTimeSeriesAnalysis:
    """
    Advanced time series analysis for energy consumption data.
    """
    
    def __init__(self, df):
        """
        Initialize analyzer with data.
        
        Args:
            df: pandas.DataFrame containing energy consumption data
        """
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
    
    def decompose_time_series(self, household_id=None, period=24):
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            household_id: Specific household to analyze (None for all)
            period: Seasonal period (24 for hourly data)
        
        Returns:
            dict: Decomposed components
        """
        if household_id:
            data = self.df[self.df['household_id'] == household_id]['consumption_kwh'].values
        else:
            data = self.df.groupby('timestamp')['consumption_kwh'].mean().values
        
        # Simple moving average for trend
        window = min(period * 7, len(data) // 10)  # 7 days or 10% of data
        if window % 2 == 0:
            window += 1
        
        trend = pd.Series(data).rolling(window=window, center=True).mean().values
        
        # Detrend
        detrended = data - trend
        
        # Seasonal component (average pattern)
        n_seasons = len(data) // period
        seasonal_pattern = []
        for i in range(period):
            indices = range(i, len(data), period)
            if len(indices) > 0:
                seasonal_pattern.append(np.mean([detrended[j] for j in indices if j < len(detrended)]))
            else:
                seasonal_pattern.append(0)
        
        # Extend seasonal pattern
        seasonal = np.tile(seasonal_pattern, n_seasons + 1)[:len(data)]
        
        # Residual
        residual = detrended - seasonal
        
        return {
            'original': data,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def calculate_autocorrelation(self, household_id=None, max_lags=48):
        """
        Calculate autocorrelation function.
        
        Args:
            household_id: Specific household to analyze
            max_lags: Maximum number of lags to calculate
        
        Returns:
            dict: Autocorrelation values
        """
        if household_id:
            data = self.df[self.df['household_id'] == household_id]['consumption_kwh'].values
        else:
            data = self.df.groupby('timestamp')['consumption_kwh'].mean().values
        
        autocorrs = {}
        for lag in range(1, min(max_lags + 1, len(data) // 2)):
            if lag < len(data):
                corr, _ = pearsonr(data[lag:], data[:-lag])
                autocorrs[lag] = corr
        
        return autocorrs
    
    def detect_trend(self, household_id=None):
        """
        Detect trend in the time series using Mann-Kendall test.
        
        Args:
            household_id: Specific household to analyze
        
        Returns:
            dict: Trend analysis results
        """
        if household_id:
            data = self.df[self.df['household_id'] == household_id]['consumption_kwh'].values
        else:
            data = self.df.groupby('timestamp')['consumption_kwh'].mean().values
        
        # Mann-Kendall test
        n = len(data)
        s = 0
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend direction
        if z > 0:
            trend_direction = "Increasing"
        elif z < 0:
            trend_direction = "Decreasing"
        else:
            trend_direction = "No trend"
        
        return {
            'z_score': z,
            'p_value': p_value,
            'trend_direction': trend_direction,
            'significant': p_value < 0.05
        }
    
    def calculate_stationarity(self, household_id=None):
        """
        Test for stationarity using Augmented Dickey-Fuller test approximation.
        
        Args:
            household_id: Specific household to analyze
        
        Returns:
            dict: Stationarity test results
        """
        if household_id:
            data = self.df[self.df['household_id'] == household_id]['consumption_kwh'].values
        else:
            data = self.df.groupby('timestamp')['consumption_kwh'].mean().values
        
        # Simple variance ratio test
        n = len(data)
        first_half = data[:n//2]
        second_half = data[n//2:]
        
        var_first = np.var(first_half)
        var_second = np.var(second_half)
        
        variance_ratio = var_second / var_first if var_first > 0 else 1
        
        # Mean difference
        mean_diff = np.abs(np.mean(second_half) - np.mean(first_half))
        mean_avg = np.mean(data)
        mean_change_pct = (mean_diff / mean_avg * 100) if mean_avg > 0 else 0
        
        # Stationary if variance ratio close to 1 and small mean change
        is_stationary = (0.8 < variance_ratio < 1.2) and (mean_change_pct < 10)
        
        return {
            'variance_ratio': variance_ratio,
            'mean_change_percent': mean_change_pct,
            'is_stationary': is_stationary
        }
    
    def calculate_seasonality_strength(self, household_id=None):
        """
        Calculate strength of seasonality.
        
        Args:
            household_id: Specific household to analyze
        
        Returns:
            dict: Seasonality metrics
        """
        if household_id:
            data = self.df[self.df['household_id'] == household_id]
        else:
            data = self.df.copy()
        
        # Group by hour and calculate variance
        hourly_variance = data.groupby('hour')['consumption_kwh'].var().mean()
        overall_variance = data['consumption_kwh'].var()
        
        # Group by day of week
        daily_variance = data.groupby('day_of_week')['consumption_kwh'].var().mean()
        
        # Group by month
        data['month'] = data['timestamp'].dt.month
        monthly_variance = data.groupby('month')['consumption_kwh'].var().mean()
        
        seasonality_strength = {
            'hourly_seasonality': hourly_variance / overall_variance if overall_variance > 0 else 0,
            'daily_seasonality': daily_variance / overall_variance if overall_variance > 0 else 0,
            'monthly_seasonality': monthly_variance / overall_variance if overall_variance > 0 else 0
        }
        
        return seasonality_strength
    
    def comprehensive_analysis(self, household_id=None):
        """
        Run comprehensive time series analysis.
        
        Args:
            household_id: Specific household to analyze
        
        Returns:
            dict: Complete analysis results
        """
        print(f"Running comprehensive analysis{' for ' + household_id if household_id else ''}...")
        
        results = {
            'decomposition': self.decompose_time_series(household_id),
            'autocorrelation': self.calculate_autocorrelation(household_id),
            'trend': self.detect_trend(household_id),
            'stationarity': self.calculate_stationarity(household_id),
            'seasonality': self.calculate_seasonality_strength(household_id)
        }
        
        return results

def main():
    """
    Main function to demonstrate advanced analysis.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - ADVANCED TIME SERIES ANALYSIS")
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
    
    # Initialize analyzer
    analyzer = AdvancedTimeSeriesAnalysis(df)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS RESULTS")
    print("=" * 60)
    
    # Trend analysis
    print("\nTREND ANALYSIS:")
    print(f"  Direction: {results['trend']['trend_direction']}")
    print(f"  Z-Score: {results['trend']['z_score']:.3f}")
    print(f"  P-Value: {results['trend']['p_value']:.4f}")
    print(f"  Significant: {'Yes' if results['trend']['significant'] else 'No'}")
    
    # Stationarity
    print("\nSTATIONARITY TEST:")
    print(f"  Variance Ratio: {results['stationarity']['variance_ratio']:.3f}")
    print(f"  Mean Change: {results['stationarity']['mean_change_percent']:.2f}%")
    print(f"  Stationary: {'Yes' if results['stationarity']['is_stationary'] else 'No'}")
    
    # Seasonality
    print("\nSEASONALITY STRENGTH:")
    print(f"  Hourly: {results['seasonality']['hourly_seasonality']:.3f}")
    print(f"  Daily: {results['seasonality']['daily_seasonality']:.3f}")
    print(f"  Monthly: {results['seasonality']['monthly_seasonality']:.3f}")
    
    # Autocorrelation (show first 10 lags)
    print("\nAUTOCORRELATION (First 10 Lags):")
    for lag in sorted(list(results['autocorrelation'].keys()))[:10]:
        print(f"  Lag {lag:2d}: {results['autocorrelation'][lag]:.3f}")
    
    print("\n" + "=" * 60)
    print("Advanced analysis complete!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

