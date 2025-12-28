"""
Energy Consumption Dataset - Data Generation Script

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script generates sample energy consumption data with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_energy_data(start_date='2023-01-01', days=365, num_households=5):
    """
    Generate synthetic energy consumption data with realistic patterns.
    
    Args:
        start_date: Start date for the dataset (YYYY-MM-DD)
        days: Number of days to generate
        num_households: Number of households
    
    Returns:
        pandas.DataFrame: Generated dataset
    """
    # Initialize lists
    data = []
    
    # Base consumption patterns
    base_consumption = {
        0: 0.8,  1: 0.7,  2: 0.6,  3: 0.6,  4: 0.7,  5: 0.8,
        6: 1.2,  7: 1.5,  8: 1.8,  9: 1.6, 10: 1.4, 11: 1.3,
        12: 1.5, 13: 1.4, 14: 1.3, 15: 1.4, 16: 1.6, 17: 2.0,
        18: 2.5, 19: 2.8, 20: 2.6, 21: 2.2, 22: 1.8, 23: 1.2
    }
    
    # Seasonal multipliers (higher in summer and winter)
    seasonal_mult = {
        1: 1.1,  2: 1.0,  3: 0.9,  4: 0.85, 5: 0.9,  6: 1.2,
        7: 1.3,  8: 1.2,  9: 1.0, 10: 0.95, 11: 1.0, 12: 1.1
    }
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    for day in range(days):
        current_date = start + timedelta(days=day)
        month = current_date.month
        day_of_week = current_date.weekday()
        
        # Weekend multiplier (slightly higher consumption)
        weekend_mult = 1.1 if day_of_week >= 5 else 1.0
        
        for hour in range(24):
            for household_id in range(1, num_households + 1):
                # Base consumption for this hour
                base = base_consumption[hour]
                
                # Apply seasonal variation
                seasonal = seasonal_mult[month]
                
                # Add random variation
                random_factor = np.random.normal(1.0, 0.15)
                
                # Household-specific variation
                household_factor = 0.8 + (household_id - 1) * 0.1
                
                # Calculate consumption
                consumption = base * seasonal * weekend_mult * random_factor * household_factor
                consumption = max(0.1, consumption)  # Ensure positive
                
                # Generate temperature (correlated with consumption)
                if month in [6, 7, 8]:  # Summer
                    temperature = np.random.normal(32, 3)
                elif month in [12, 1, 2]:  # Winter
                    temperature = np.random.normal(15, 3)
                else:  # Spring/Fall
                    temperature = np.random.normal(22, 4)
                
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                
                data.append({
                    'timestamp': timestamp,
                    'household_id': f'HH{household_id:03d}',
                    'consumption_kwh': round(consumption, 2),
                    'temperature': round(temperature, 1),
                    'hour': hour,
                    'day_of_week': day_of_week
                })
    
    df = pd.DataFrame(data)
    return df

def save_to_csv(df, filename='energy_consumption.csv'):
    """
    Save dataframe to CSV file.
    
    Args:
        df: pandas.DataFrame to save
        filename: Output filename
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Total records: {len(df)}")

def save_to_json(df, filename='energy_consumption.json'):
    """
    Save dataframe to JSON file.
    
    Args:
        df: pandas.DataFrame to save
        filename: Output filename
    """
    # Convert timestamp to string for JSON
    df_json = df.copy()
    df_json['timestamp'] = df_json['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert to records format
    records = df_json.to_dict('records')
    
    with open(filename, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Data saved to {filename}")
    print(f"Total records: {len(records)}")

def main():
    """
    Main function to generate and save energy consumption data.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - DATA GENERATION")
    print("=" * 60)
    print("Project: Energy Consumption Dataset")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("=" * 60 + "\n")
    
    print("Generating energy consumption data...")
    print("This may take a few moments...\n")
    
    # Generate data
    df = generate_energy_data(start_date='2023-01-01', days=365, num_households=5)
    
    # Save to CSV
    save_to_csv(df, 'energy_consumption.csv')
    
    # Save to JSON
    save_to_json(df, 'energy_consumption.json')
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Total Records: {len(df):,}")
    print(f"  Number of Households: {df['household_id'].nunique()}")
    print(f"  Average Consumption: {df['consumption_kwh'].mean():.2f} kWh")
    print(f"\nFor more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

