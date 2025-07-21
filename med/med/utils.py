"""Utilities for post-processing experiment results."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def export_and_plot_results(strategy, experiment_name: str, output_base_dir: str = "research_exports"):
    """Export research data and create plots after simulation completes."""
    try:
        # 1. Xuất dữ liệu nghiên cứu
        print("\nSimulation finished. Exporting research data...")
        output_directory = os.path.join(output_base_dir, experiment_name)
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Export research data if strategy supports it
        if hasattr(strategy, 'export_research_data'):
            strategy.export_research_data(output_dir=output_directory)
            print(f"Data exported to: {output_directory}")
        else:
            print("Strategy doesn't support export_research_data method")
            return
        
        # 2. Vẽ biểu đồ kết quả
        plot_experiment_results(output_directory, experiment_name)
        
    except Exception as e:
        print(f"Error in post-processing: {e}")


def plot_experiment_results(output_directory: str, experiment_name: str):
    """Create plots from exported experiment data."""
    try:
        server_metrics_file = os.path.join(output_directory, "server_round_data.csv")
        
        if not os.path.exists(server_metrics_file):
            print(f"Server metrics file not found: {server_metrics_file}")
            return
            
        server_metrics_df = pd.read_csv(server_metrics_file)
        
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.set_xlabel('Round')
        ax1.set_ylabel('AdaFedAdam Metrics', color='tab:blue')
        
        # Plot variance and adapted eta if available
        if 'variance' in server_metrics_df.columns:
            ax1.plot(server_metrics_df['round'], server_metrics_df['variance'], 'b-', label='Variance of Updates')
        if 'adapted_eta' in server_metrics_df.columns:
            ax1.plot(server_metrics_df['round'], server_metrics_df['adapted_eta'], 'c--', label='Adapted Eta (η)')
        
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, linestyle=':')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (Avg Dice)', color='tab:red')
        
        # Plot accuracy metrics if available
        if 'server_accuracy' in server_metrics_df.columns:
            ax2.plot(server_metrics_df['round'], server_metrics_df['server_accuracy'], 'r-', label='Server Accuracy')
        
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title(f'Experiment: {experiment_name} - Server Metrics')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            ax2.legend(lines + lines2, labels + labels2, loc='best')
        
        plot_path = os.path.join(output_directory, "server_metrics_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        
        # Don't show plot in non-interactive environments
        # plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Could not plot results: {e}")


def create_experiment_summary(output_directory: str, experiment_name: str, config: dict):
    """Create a summary file for the experiment."""
    try:
        summary = {
            "experiment_name": experiment_name,
            "configuration": config,
            "output_directory": output_directory
        }
        
        summary_file = os.path.join(output_directory, "experiment_summary.json")
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Experiment summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Could not create experiment summary: {e}") 