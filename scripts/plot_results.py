#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import argparse

def collect_all_results(base_path="../my_results"):
    """
    Collect all progress.csv files from the experiment results
    """
    results = []
    
    # Find all progress.csv files
    pattern = os.path.join(base_path, "**", "progress.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(csv_files)} progress.csv files")
    
    for csv_file in csv_files:
        try:
            # Extract algorithm and environment from path
            path_parts = Path(csv_file).parts
            
            # Find the algorithm (ql, ql-rs, crm, crm-rs, hrm, hrm-rs)
            alg_idx = None
            env_name = None
            
            for i, part in enumerate(path_parts):
                if part in ['ql', 'ql-rs', 'crm', 'crm-rs', 'hrm', 'hrm-rs']:
                    alg_idx = i
                    algorithm = part
                    # Next part should be environment
                    if i + 1 < len(path_parts):
                        env_name = path_parts[i + 1]
                    break
            
            if alg_idx is None:
                print(f"Could not determine algorithm for {csv_file}")
                continue
                
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add metadata
            df['algorithm'] = algorithm
            df['environment'] = env_name
            df['file_path'] = csv_file
            
            # Extract trial number from path
            trial_num = None
            for part in path_parts:
                if part.isdigit():
                    trial_num = int(part)
                    break
            df['trial'] = trial_num if trial_num is not None else 0
            
            results.append(df)
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

def categorize_columns(df):
    """
    Automatically categorize columns by their type and content
    """
    # Exclude metadata columns
    excluded_cols = {'algorithm', 'environment', 'file_path', 'trial'}
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
    
    categories = {
        'x_axis': [],
        'rewards': [],
        'rates': [],
        'counts': [],
        'other': []
    }
    
    for col in numeric_cols:
        col_lower = col.lower()
        
        # X-axis column (typically steps)
        if 'step' in col_lower:
            categories['x_axis'].append(col)
        # Reward columns
        elif 'reward' in col_lower:
            categories['rewards'].append(col)
        # Rate/success columns (should be between 0 and 1)
        elif any(word in col_lower for word in ['rate', 'success', 'efficiency']):
            categories['rates'].append(col)
        # Count columns
        elif any(word in col_lower for word in ['episode', 'count', 'number']):
            categories['counts'].append(col)
        else:
            categories['other'].append(col)
    
    return categories

def plot_generic_learning_curves(df, columns, save_path, plot_name, ylabel, ylim=None):
    """
    Generic function to plot learning curves for any set of columns
    """
    if not columns:
        print(f"Warning: No columns found for {plot_name}. Skipping.")
        return
    
    # Determine x-axis column (prefer 'steps', fallback to first available)
    x_col = 'steps' if 'steps' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    
    # Set up the plotting style
    plt.style.use('default')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Get unique environments
    environments = df['environment'].unique()
    
    # Create subplots for each column
    n_cols = len(columns)
    n_envs = len(environments)
    
    fig, axes = plt.subplots(n_envs, n_cols, figsize=(6*n_cols, 6*n_envs))
    if n_envs == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_envs == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for env_idx, env in enumerate(environments):
        env_data = df[df['environment'] == env]
        
        for col_idx, col in enumerate(columns):
            ax = axes[env_idx][col_idx]
            
            # Plot each algorithm
            algorithms = env_data['algorithm'].unique()
            
            for alg_idx, alg in enumerate(algorithms):
                alg_data = env_data[env_data['algorithm'] == alg]
                color = colors[alg_idx % len(colors)]
                
                # Filter out NaN values
                alg_data_clean = alg_data.dropna(subset=[col])
                
                if alg_data_clean.empty:
                    continue
                
                # Group by x-axis and calculate mean and std across trials
                grouped = alg_data_clean.groupby(x_col)[col].agg(['mean', 'std', 'count']).reset_index()
                
                # Calculate confidence interval
                confidence_interval = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
                confidence_interval = confidence_interval.fillna(0)
                
                # Plot mean line
                ax.plot(grouped[x_col], grouped['mean'], label=alg, linewidth=2, color=color)
                
                # Plot confidence interval
                ax.fill_between(grouped[x_col], 
                               grouped['mean'] - confidence_interval,
                               grouped['mean'] + confidence_interval,
                               alpha=0.2, color=color)
            
            ax.set_xlabel(x_col.title())
            ax.set_ylabel(ylabel)
            ax.set_title(f'{col} - {env}')
            if ylim:
                ax.set_ylim(ylim)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{plot_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_comparison(df, columns, save_path, plot_name, ylabel):
    """
    Generic function to create final performance comparison box plots
    """
    if not columns:
        print(f"Warning: No columns found for {plot_name}. Skipping.")
        return
    
    for col in columns:
        # Get the final value for each trial
        final_values = df.dropna(subset=[col]).groupby(['algorithm', 'environment', 'trial'])[col].last().reset_index()
        
        if final_values.empty:
            continue
        
        # Get unique algorithms and environments
        algorithms = final_values['algorithm'].unique()
        environments = final_values['environment'].unique()
        
        plt.figure(figsize=(12, 6))
        
        # Prepare data for box plot
        box_data = []
        labels = []
        colors = plt.cm.Set1(np.linspace(0, 1, len(environments)))
        
        x_positions = []
        current_pos = 0
        
        for i, alg in enumerate(algorithms):
            for j, env in enumerate(environments):
                subset = final_values[(final_values['algorithm'] == alg) & 
                                    (final_values['environment'] == env)]
                if not subset.empty:
                    box_data.append(subset[col].values)
                    labels.append(f"{alg}\n{env}")
                    x_positions.append(current_pos)
                    current_pos += 1
            current_pos += 0.5  # Add space between algorithm groups
        
        if not box_data:
            continue
        
        # Create box plot
        bp = plt.boxplot(box_data, positions=x_positions, patch_artist=True, widths=0.6)
        
        # Color the boxes by environment
        for i, (patch, label) in enumerate(zip(bp['boxes'], labels)):
            env = label.split('\n')[1]
            env_idx = np.where(environments == env)[0][0]
            patch.set_facecolor(colors[env_idx])
            patch.set_alpha(0.7)
        
        plt.xticks(x_positions, labels, rotation=45, ha='right')
        plt.title(f'Final {col} Comparison')
        plt.xlabel('Algorithm / Environment')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        # Create custom legend for environments
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=env) 
                          for i, env in enumerate(environments)]
        plt.legend(handles=legend_elements, title='Environment', loc='upper right')
        
        plt.tight_layout()
        safe_col_name = col.replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(save_path, f'final_{safe_col_name}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

def print_summary_stats(df, categories):
    """
    Print comprehensive summary statistics for all numeric columns
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*60)
    
    # All numeric columns (excluding metadata)
    excluded_cols = {'algorithm', 'environment', 'file_path', 'trial'}
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in excluded_cols]
    
    # Final performance statistics for all numeric columns
    for col in numeric_cols:
        final_values = df.dropna(subset=[col]).groupby(['algorithm', 'environment', 'trial'])[col].last().reset_index()
        
        if final_values.empty:
            continue
        
        summary = final_values.groupby(['algorithm', 'environment'])[col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        print(f"\nFinal {col} Summary:")
        print(summary)
    
    # Sample efficiency analysis for reward columns
    if categories['rewards']:
        print(f"\n" + "-"*60)
        print("Sample Efficiency Analysis (Rewards)")
        print("-"*60)
        
        x_col = 'steps' if 'steps' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        
        for col in categories['rewards']:
            # Determine appropriate thresholds based on data
            col_data = df[col].dropna()
            if col_data.empty:
                continue
                
            max_val = col_data.max()
            thresholds = [max_val * 0.25, max_val * 0.5, max_val * 0.75, max_val * 0.9]
            thresholds = [t for t in thresholds if t > 0]
            
            for threshold in thresholds:
                print(f"\nSteps to reach {threshold:.2f} {col}:")
                for alg in df['algorithm'].unique():
                    for env in df['environment'].unique():
                        subset = df[(df['algorithm'] == alg) & (df['environment'] == env)]
                        subset_clean = subset.dropna(subset=[col])
                        
                        if subset_clean.empty:
                            continue
                        
                        steps_to_threshold = []
                        for trial in subset_clean['trial'].unique():
                            trial_data = subset_clean[subset_clean['trial'] == trial]
                            reached = trial_data[trial_data[col] >= threshold]
                            if not reached.empty:
                                steps_to_threshold.append(reached[x_col].iloc[0])
                        
                        if steps_to_threshold:
                            mean_steps = np.mean(steps_to_threshold)
                            success_rate = len(steps_to_threshold) / len(subset_clean['trial'].unique())
                            print(f"  {alg:>8} on {env:>12}: {mean_steps:>8.0f} steps ({success_rate:>5.1%} success)")

def main():
    parser = argparse.ArgumentParser(description='Plot experiment results automatically.')
    parser.add_argument('--base_path', type=str, default="../my_results", help='Base path for experiment results')
    args = parser.parse_args()
    
    print("Collecting experiment results...")
    df = collect_all_results(base_path=args.base_path)
    
    if df.empty:
        print("No data found! Make sure you're running this from the scripts directory.")
        return
    
    print(f"Loaded data from {len(df)} rows across {len(df['trial'].unique())} trials")
    print(f"Algorithms: {', '.join(df['algorithm'].unique())}")
    print(f"Environments: {', '.join(df['environment'].unique())}")
    
    # Automatically categorize columns
    categories = categorize_columns(df)
    
    print(f"\nDetected columns:")
    for category, cols in categories.items():
        if cols:
            print(f"  {category}: {cols}")
    
    # Generate plots for each category
    print("\nGenerating plots...")
    
    # Reward curves
    if categories['rewards']:
        print("Generating reward learning curves...")
        plot_generic_learning_curves(df, categories['rewards'], args.base_path, 
                                    'reward_curves', 'Reward')
    
    # Rate/success curves (0-1 scale)
    if categories['rates']:
        print("Generating rate/success learning curves...")
        plot_generic_learning_curves(df, categories['rates'], args.base_path, 
                                    'rate_curves', 'Rate/Success', ylim=(0, 1.05))
    
    # Count curves
    if categories['counts']:
        print("Generating count learning curves...")
        plot_generic_learning_curves(df, categories['counts'], args.base_path, 
                                    'count_curves', 'Count')
    
    # Other metric curves
    if categories['other']:
        print("Generating other metric learning curves...")
        plot_generic_learning_curves(df, categories['other'], args.base_path, 
                                    'other_curves', 'Value')
    
    # Final comparison plots
    print("Generating final performance comparisons...")
    
    all_metric_cols = categories['rewards'] + categories['rates'] + categories['counts'] + categories['other']
    plot_final_comparison(df, all_metric_cols, args.base_path, 'final_comparison', 'Value')
    
    # Print summary statistics
    print_summary_stats(df, categories)
    
    # List generated plots
    print(f"\nGenerated plots:")
    for plot_type in ['reward_curves', 'rate_curves', 'count_curves', 'other_curves']:
        plot_file = os.path.join(args.base_path, f'{plot_type}.png')
        if os.path.exists(plot_file):
            print(f"  - {plot_type}.png")
    
    # List final comparison plots
    all_cols = categories['rewards'] + categories['rates'] + categories['counts'] + categories['other']
    for col in all_cols:
        safe_col_name = col.replace(' ', '_').replace('/', '_')
        plot_file = os.path.join(args.base_path, f'final_{safe_col_name}_comparison.png')
        if os.path.exists(plot_file):
            print(f"  - final_{safe_col_name}_comparison.png")

if __name__ == "__main__":
    main() 