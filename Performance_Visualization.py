"""
Performance Visualization: Reinforcement Learning - Efficiency vs Survival Comparison
Includes DQN 2-Layer (Runs 1-5) and DQN 3-Layer (Runs 6-10) with Plots and Tables
@author: Weifu Lin
Date: 8/9/2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# 1. Setup Parameters & Constants
window_size = 100
base_save_dir = r"C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\Performance_Plots_&_Matrices"

# Ensure output directory exists to avoid save errors
os.makedirs(base_save_dir, exist_ok=True)


# Path validation helper function
def get_valid_path(primary_path):
    if os.path.exists(primary_path):
        return primary_path

    folder = os.path.dirname(primary_path)
    if os.path.exists(folder):
        # Look for any .csv file inside the run directory as a fallback
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if csv_files:
            return os.path.join(folder, csv_files[0])

    raise FileNotFoundError(f"No valid CSV file found in folder: {folder}")


# Paths setup using list comprehensions
random_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults' f'\\{i}\\random_agent_stats.csv'
    for i in range(1, 6)]
qlearn_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\QLearningResults' f'\\{i}\\qlearn-stats.csv' for i in
    range(1, 6)]
dqn_2layer_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\2 Layer' f'\\{i}\\dqnlearn-stats.csv' for
    i in range(1, 6)]
dqn_3layer_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\3 Layer' f'\\{i}\\dqnlearn-stats.csv' for
    i in range(1, 6)]

metrics = [
    ('Score', 'Efficiency (Score)'),
    ('Total Steps', 'Survival Capability (Steps)')
]

font_legend = font_manager.FontProperties(family='Arial', size=22, style='normal')


# Function to Save DataFrame as Table and PNG
def save_dataframe_as_image(df, save_path, title="Metrics Summary"):
    fig, ax = plt.subplots(figsize=(10, max(2, len(df) * 0.8 + 0.5)))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.8)

    # Header and row setting
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#7494BC')  # Header Color setting
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#BFE0F9' if row % 2 == 0 else '#ffffff')

    plt.title(title, fontsize=18, fontweight='bold', pad=20, color='#1B4D3E')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# 2. Main Processing Function
def process_run(run_folder_num, random_csv, qlearn_csv, dqn_csv, dqn_label):
    target_save_dir = os.path.join(base_save_dir, str(run_folder_num))
    os.makedirs(target_save_dir, exist_ok=True)

    models = {
        'Random Agent': {'path': get_valid_path(random_csv), 'raw_color': '#d3d3d3', 'ma_color': '#555555'},
        'Q-Learning': {'path': get_valid_path(qlearn_csv), 'raw_color': '#eb8888', 'ma_color': '#e64c56'},
        dqn_label: {'path': get_valid_path(dqn_csv), 'raw_color': '#27ecec', 'ma_color': '#276cec'}
    }

    dfs = {}
    for name, info in models.items():
        df = pd.read_csv(info['path'])

        # Coerce 'Episode' column to numeric to fix string vs int NumPy comparisons
        if 'Episode' in df.columns:
            df['Episode'] = pd.to_numeric(df['Episode'], errors='coerce')

        # Coerce target metrics to numeric values
        for col_name, _ in metrics:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        # Drop corrupted header/NaN rows
        dfs[name] = df.dropna(subset=['Episode']).copy()

    # Plot Performance Comparison
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
    axes = [ax0, ax1]

    max_episodes = 0

    for ax, (col, ylabel) in zip(axes, metrics):
        for name, info in models.items():
            df = dfs[name]

            # Track maximum episode count dynamically across all models
            if not df.empty:
                max_episodes = max(max_episodes, int(df['Episode'].max()))

            ma = df[col].rolling(window=window_size, min_periods=1).mean()
            ax.plot(df['Episode'], df[col], color=info['raw_color'], alpha=0.15, label=f'{name} (Raw)')
            ax.plot(df['Episode'], ma, color=info['ma_color'], linewidth=3.5, label=f'{name} ({window_size}-Ep MA)')

        ax.set_ylabel(ylabel, fontname="Arial", fontsize=28, rotation='vertical')
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.grid(True, linestyle='--', alpha=0.3, color='black')

        leg = ax.legend(prop=font_legend, borderaxespad=0, labelspacing=0.4, handlelength=1.5, frameon=False,
                        loc='upper left')
        for line in leg.get_lines():
            line.set_linewidth(4)

    axes[1].set_xlabel('Episode $i$', fontsize=30, fontname='Arial')
    axes[1].set_xlim([1, max(1, max_episodes)])  # Dynamic X-axis bound check

    plt.tight_layout()

    filename = f"RL_Efficiency_vs_Survival_Comparison_{run_folder_num}.png"
    plt.savefig(os.path.join(target_save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Metrics Summary Calculation
    summary_data = []
    for name, df in dfs.items():
        summary_data.append({
            'Run ID': run_folder_num,
            'Agent': name,
            'Mean Score': round(df['Score'].mean(), 2),
            'Max Score': df['Score'].max(),
            'Mean Steps': round(df['Total Steps'].mean(), 2),
            'Max Steps': df['Total Steps'].max()
        })

    summary_df = pd.DataFrame(summary_data)

    # Save individual CSV table
    summary_df.to_csv(os.path.join(target_save_dir, f"Performance_Summary_{run_folder_num}.csv"), index=False)

    # Save individual PNG image table
    table_img_path = os.path.join(target_save_dir, f"Summary_Table_Image_{run_folder_num}.png")
    save_dataframe_as_image(summary_df, table_img_path, title=f"Run {run_folder_num} Metrics Summary")

    return summary_data


# 3. Execution Loops
all_master_summaries = []

# Part A: Runs 1 to 5 (DQN 2 Layer vs Q-Learning vs Random)
print("Processing Runs 1 to 5 (DQN 2 Layer vs Q-Learning vs Random)...")
for idx in range(5):
    folder_num = idx + 1
    run_summary = process_run(folder_num, random_paths[idx], qlearn_paths[idx], dqn_2layer_paths[idx], 'DQN (2 Layer)')
    all_master_summaries.extend(run_summary)

# Part B: Runs 6 to 10 (DQN 3 Layer vs Q-Learning vs Random)
print("Processing Runs 6 to 10 (DQN 3 Layer vs Q-Learning vs Random)...")
for idx in range(5):
    folder_num = idx + 6
    run_summary = process_run(folder_num, random_paths[idx], qlearn_paths[idx], dqn_3layer_paths[idx], 'DQN (3 Layer)')
    all_master_summaries.extend(run_summary)

# Save master Summary CSV and PNG table
master_df = pd.DataFrame(all_master_summaries)
master_csv_path = os.path.join(base_save_dir, "Master_Performance_Summary_All_Runs.csv")
master_df.to_csv(master_csv_path, index=False)

master_img_path = os.path.join(base_save_dir, "Master_Performance_Summary_Table.png")
save_dataframe_as_image(master_df, master_img_path, title="Master Performance Summary (All 10 Runs)")

print(f"\nSuccessfully generated charts, CSV tables, and PNG image tables!")
print(f"Master summary image saved to: {master_img_path}")