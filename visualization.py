"""
Energy Consumption Dataset - Visualization Script

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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

def plot_time_series(df, save_path='time_series_plot.png'):
    """
    Plot time series of energy consumption.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    # Plot consumption over time
    df_sorted = df.sort_values('timestamp')
    plt.plot(df_sorted['timestamp'], df_sorted['consumption_kwh'], 
             linewidth=0.5, alpha=0.7, color='#2ecc71')
    plt.title('Energy Consumption Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Consumption (kWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved to {save_path}")
    plt.close()

def plot_hourly_patterns(df, save_path='hourly_patterns.png'):
    """
    Plot average consumption by hour of day.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    hourly_avg = df.groupby('hour')['consumption_kwh'].mean()
    
    plt.bar(hourly_avg.index, hourly_avg.values, color='#3498db', alpha=0.7)
    plt.title('Average Energy Consumption by Hour of Day', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Consumption (kWh)', fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Hourly patterns plot saved to {save_path}")
    plt.close()

def plot_seasonal_patterns(df, save_path='seasonal_patterns.png'):
    """
    Plot seasonal consumption patterns.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    df['month'] = df['timestamp'].dt.month
    monthly_avg = df.groupby('month')['consumption_kwh'].mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.plot(monthly_avg.index, monthly_avg.values, 
             marker='o', linewidth=2, markersize=8, color='#e74c3c')
    plt.title('Average Energy Consumption by Month', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Consumption (kWh)', fontsize=12)
    plt.xticks(monthly_avg.index, [months[i-1] for i in monthly_avg.index])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Seasonal patterns plot saved to {save_path}")
    plt.close()

def plot_household_comparison(df, save_path='household_comparison.png'):
    """
    Plot consumption comparison across households.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    household_avg = df.groupby('household_id')['consumption_kwh'].mean().sort_values()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(household_avg)))
    plt.barh(range(len(household_avg)), household_avg.values, color=colors)
    plt.yticks(range(len(household_avg)), household_avg.index)
    plt.title('Average Energy Consumption by Household', fontsize=16, fontweight='bold')
    plt.xlabel('Average Consumption (kWh)', fontsize=12)
    plt.ylabel('Household ID', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Household comparison plot saved to {save_path}")
    plt.close()

def plot_heatmap(df, save_path='consumption_heatmap.png'):
    """
    Create heatmap of consumption by hour and day of week.
    
    Args:
        df: pandas.DataFrame containing energy consumption data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values='consumption_kwh', 
        index='day_of_week', 
        columns='hour', 
        aggfunc='mean'
    )
    
    # Map day numbers to names
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_data.index = [day_names[i] for i in pivot_data.index]
    
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, fmt='.1f', 
               cbar_kws={'label': 'Consumption (kWh)'})
    plt.title('Energy Consumption Heatmap: Day of Week vs Hour', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    plt.close()

def main():
    """
    Main function to generate all visualizations.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - VISUALIZATION")
    print("=" * 60)
    print("Project: Energy Consumption Dataset")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("=" * 60 + "\n")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Generate visualizations
    print("Generating visualizations...\n")
    plot_time_series(df)
    plot_hourly_patterns(df)
    plot_seasonal_patterns(df)
    plot_household_comparison(df)
    plot_heatmap(df)
    
    print("\nAll visualizations generated successfully!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

