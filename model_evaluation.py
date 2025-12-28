"""
Energy Consumption Dataset - Model Evaluation and Metrics

Project: Energy Consumption Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Comprehensive model evaluation metrics and comparison utilities.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Model evaluation class for comparing forecasting models.
    """
    
    def __init__(self):
        """
        Initialize evaluator.
        """
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name='Model'):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
        
        Returns:
            dict: Evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Mean Bias Error
        mbe = np.mean(y_pred - y_true)
        
        # Coefficient of Variation of RMSE
        cv_rmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) > 0 else 0
        
        metrics = {
            'model_name': model_name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R²': round(r2, 4),
            'MAPE': round(mape, 2),
            'MBE': round(mbe, 4),
            'CV(RMSE)': round(cv_rmse, 2)
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return metrics
    
    def compare_models(self):
        """
        Compare all evaluated models.
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.results:
            print("No models evaluated yet.")
            return None
        
        comparison = []
        for model_name, result in self.results.items():
            comparison.append(result['metrics'])
        
        df = pd.DataFrame(comparison)
        return df
    
    def plot_predictions(self, model_name, save_path=None):
        """
        Plot predictions vs actual values.
        
        Args:
            model_name: Name of the model to plot
            save_path: Path to save the plot
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found.")
            return
        
        result = self.results[model_name]
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Consumption (kWh)')
        plt.ylabel('Predicted Consumption (kWh)')
        plt.title(f'{model_name} - Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Consumption (kWh)')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison(self, save_path=None):
        """
        Plot comparison of all models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No models to compare.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(self.results.keys())
        metrics_to_plot = ['MAE', 'RMSE', 'R²', 'MAPE']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[model]['metrics'][metric] for model in models]
            
            bars = ax.bar(models, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(models)])
            ax.set_ylabel(metric)
            ax.set_title(f'Model Comparison - {metric}')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, save_path='model_evaluation_report.txt'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("Project: Energy Consumption Dataset")
        report.append("Author: RSK World")
        report.append("Website: https://rskworld.in")
        report.append("=" * 60)
        report.append("")
        
        if not self.results:
            report.append("No models evaluated yet.")
        else:
            comparison = self.compare_models()
            report.append("MODEL COMPARISON")
            report.append("-" * 60)
            report.append(comparison.to_string(index=False))
            report.append("")
            
            report.append("METRIC DESCRIPTIONS")
            report.append("-" * 60)
            report.append("MAE (Mean Absolute Error): Average absolute difference")
            report.append("RMSE (Root Mean Squared Error): Penalizes larger errors")
            report.append("R² (R-squared): Proportion of variance explained")
            report.append("MAPE (Mean Absolute Percentage Error): Percentage error")
            report.append("MBE (Mean Bias Error): Average prediction bias")
            report.append("CV(RMSE): Coefficient of variation of RMSE")
            report.append("")
            
            # Best model for each metric
            report.append("BEST MODELS BY METRIC")
            report.append("-" * 60)
            for metric in ['MAE', 'RMSE', 'R²', 'MAPE']:
                if metric == 'R²':
                    best_model = comparison.loc[comparison[metric].idxmax(), 'model_name']
                    best_value = comparison[metric].max()
                else:
                    best_model = comparison.loc[comparison[metric].idxmin(), 'model_name']
                    best_value = comparison[metric].min()
                report.append(f"{metric}: {best_model} ({best_value:.4f})")
        
        report.append("")
        report.append("=" * 60)
        report.append("For more information, visit: https://rskworld.in")
        
        report_text = "\n".join(report)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {save_path}")

def main():
    """
    Main function to demonstrate model evaluation.
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION DATASET - MODEL EVALUATION")
    print("=" * 60)
    print("Project: Energy Consumption Dataset")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("=" * 60 + "\n")
    
    # Example usage with dummy data
    evaluator = ModelEvaluator()
    
    # Generate example predictions (in real usage, these would come from models)
    np.random.seed(42)
    y_true = np.random.uniform(0.5, 3.0, 1000)
    y_pred_lr = y_true + np.random.normal(0, 0.2, 1000)
    y_pred_rf = y_true + np.random.normal(0, 0.15, 1000)
    
    # Evaluate models
    print("Evaluating models...\n")
    evaluator.calculate_metrics(y_true, y_pred_lr, 'Linear Regression')
    evaluator.calculate_metrics(y_true, y_pred_rf, 'Random Forest')
    
    # Compare models
    comparison = evaluator.compare_models()
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    evaluator.plot_predictions('Linear Regression', 'lr_evaluation.png')
    evaluator.plot_predictions('Random Forest', 'rf_evaluation.png')
    evaluator.plot_comparison('model_comparison.png')
    
    # Generate report
    evaluator.generate_report()
    
    print("\n" + "=" * 60)
    print("Model evaluation complete!")
    print("For more information, visit: https://rskworld.in")

if __name__ == "__main__":
    main()

