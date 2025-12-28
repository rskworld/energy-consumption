"""
Energy Consumption Dataset - Data Analysis Script

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def load_data(csv_file='energy_consumption.csv'):
    """
    Load energy consumption data from CSV file.
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please ensure the file exists.")
        return None

def basic_statistics(df):
    """
    Calculate basic statistics for the dataset.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
    """
    print("=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(f"\nTotal Records: {len(df)}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of Households: {df['household_id'].nunique()}")
    print(f"\nConsumption Statistics (kWh):")
    print(df['consumption_kwh'].describe())
    print("\n" + "=" * 60)

def peak_hour_analysis(df):
    """
    Identify peak consumption hours.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
    """
    print("\nPEAK HOUR ANALYSIS")
    print("=" * 60)
    hourly_avg = df.groupby('hour')['consumption_kwh'].mean().sort_values(ascending=False)
    print("\nTop 5 Peak Hours (Average Consumption):")
    for hour, consumption in hourly_avg.head(5).items():
        print(f"  Hour {hour:02d}:00 - {consumption:.2f} kWh")
    print("\n" + "=" * 60)

def seasonal_patterns(df):
    """
    Analyze seasonal consumption patterns.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
    """
    print("\nSEASONAL PATTERNS")
    print("=" * 60)
    df['month'] = df['timestamp'].dt.month
    monthly_avg = df.groupby('month')['consumption_kwh'].mean()
    print("\nAverage Consumption by Month:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, consumption in monthly_avg.items():
        print(f"  {months[month-1]}: {consumption:.2f} kWh")
    print("\n" + "=" * 60)

def household_comparison(df):
    """
    Compare consumption across different households.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
    """
    print("\nHOUSEHOLD COMPARISON")
    print("=" * 60)
    household_stats = df.groupby('household_id')['consumption_kwh'].agg([
        'mean', 'sum', 'std'
    ]).round(2)
    household_stats.columns = ['Avg Consumption (kWh)', 'Total Consumption (kWh)', 'Std Dev']
    print("\nHousehold Statistics:")
    print(household_stats)
    print("\n" + "=" * 60)

def correlation_analysis(df):
    """
    Analyze correlation between consumption and temperature.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
    """
    print("\nCORRELATION ANALYSIS")
    print("=" * 60)
    if 'temperature' in df.columns:
        correlation = df['consumption_kwh'].corr(df['temperature'])
        print(f"\nCorrelation between Consumption and Temperature: {correlation:.3f}")
        if abs(correlation) > 0.5:
            print("  Strong correlation detected!")
        elif abs(correlation) > 0.3:
            print("  Moderate correlation detected.")
        else:
            print("  Weak correlation.")
    print("\n" + "=" * 60)

def main():
    """
    Main function to run all analyses.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - ANALYSIS")
    print("=" * 60)
    print("Project: Energy Consumption Dataset")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("=" * 60 + "\n")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run analyses
    basic_statistics(df)
    peak_hour_analysis(df)
    seasonal_patterns(df)
    household_comparison(df)
    correlation_analysis(df)
    
    print("\nAnalysis complete!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

