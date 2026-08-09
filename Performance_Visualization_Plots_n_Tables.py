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


# Finding valid CSV path (checking if file name is the same)
def get_valid_path(primary_path):
    if os.path.exists(primary_path):
        return primary_path
    folder = os.path.dirname(primary_path)
    alt_name = "dqnlearn-stats.csv" if "1layer" in primary_path else "dqnlearn-stats-1layer.csv"
    alt_path = os.path.join(folder, alt_name)
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"Neither primary nor fallback CSV file found in: {folder}")


# Define file path lists for Random Agent (reused for runs 1-5 and 6-10)
random_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults\1\random_agent_stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults\2\random_agent_stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults\3\random_agent_stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults\4\random_agent_stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults\5\random_agent_stats.csv'
]

# Define file path lists for Q-Learning (reused for runs 1-5 and 6-10)
qlearn_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\QLearningResults\1\qlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\QLearningResults\2\qlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\QLearningResults\3\qlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\QLearningResults\4\qlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\QLearningResults\5\qlearn-stats.csv'
]

# DQN 2-Layer paths (For Runs 1-5)
dqn_2layer_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\2 Layer\1\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\2 Layer\2\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\2 Layer\3\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\2 Layer\4\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\2 Layer\5\dqnlearn-stats.csv'
]

# DQN 3-Layer paths (For Runs 6-10)
dqn_3layer_paths = [
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\3 Layer\1\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\3 Layer\2\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\3 Layer\3\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\3 Layer\4\dqnlearn-stats.csv',
    r'C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\DQNResults\3 Layer\5\dqnlearn-stats.csv'
]

metrics = [
    ('Score', 'Efficiency (Score)'),
    ('Total Steps', 'Survival Capability (Steps)')
]

font_legend = font_manager.FontProperties(family='Arial', size=22, style='normal')


# Function to Save DataFrame as Table and PNG
def save_dataframe_as_image(df, save_path, title="Metrics Summary"):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.8 + 0.1))
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
            # Alternating light surf green and white rows
            cell.set_facecolor('#BFE0F9' if row % 2 == 0 else '#ffffff')

    plt.title(title, fontsize=18, fontweight='bold', pad=20, color='#1B4D3E')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# 2. Main Function
def process_run(run_folder_num, random_csv, qlearn_csv, dqn_csv, dqn_label):
    target_save_dir = os.path.join(base_save_dir, str(run_folder_num))
    os.makedirs(target_save_dir, exist_ok=True)

    models = {
        'Random Agent': {'path': random_csv, 'raw_color': '#d3d3d3', 'ma_color': '#555555'},
        'Q-Learning': {'path': qlearn_csv, 'raw_color': '#eb8888', 'ma_color': '#e64c56'},
        dqn_label: {'path': dqn_csv, 'raw_color': '#27ecec', 'ma_color': '#276cec'}
    }

    dfs = {name: pd.read_csv(info['path']) for name, info in models.items()}

    # Plot Performance Comparison
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
    axes = [ax0, ax1]

    for ax, (col, ylabel) in zip(axes, metrics):
        for name, info in models.items():
            df = dfs[name]
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

        leg = ax.legend(prop=font_legend, borderaxespad=0, labelspacing=0.4, handlelength=1.5, frameon=False, loc='upper left')
        for line in leg.get_lines():
            line.set_linewidth(4)

    axes[1].set_xlabel('Episode $i$', fontsize=30, fontname='Arial')
    axes[1].set_xlim([1, 10000])

    plt.tight_layout()

    filename = f"RL_Efficiency_vs_Survival_Comparison_{run_folder_num}.png"
    plt.savefig(os.path.join(target_save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Generate metrics and save them in CSV and PNG formats
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
print("Runs 1 to 5 (DQN 2 Layer vs Q-Learning vs Random)")
for idx in range(5):
    folder_num = idx + 1
    dqn_path = get_valid_path(dqn_2layer_paths[idx])
    run_summary = process_run(folder_num, random_paths[idx], qlearn_paths[idx], dqn_path, 'DQN (2 Layer)')
    all_master_summaries.extend(run_summary)

# Part B: Runs 6 to 10 (DQN 3 Layer vs Q-Learning vs Random)
print("Runs 6 to 10 (DQN 3 Layer vs Q-Learning vs Random)")
for idx in range(5):
    folder_num = idx + 6
    dqn_path = get_valid_path(dqn_3layer_paths[idx])
    run_summary = process_run(folder_num, random_paths[idx], qlearn_paths[idx], dqn_path, 'DQN (3 Layer)')
    all_master_summaries.extend(run_summary)

# Save master Summary of csv files and tables (optional)
master_df = pd.DataFrame(all_master_summaries)
master_csv_path = os.path.join(base_save_dir, "Master_Performance_Summary_All_Runs.csv")
master_df.to_csv(master_csv_path, index=False)

master_img_path = os.path.join(base_save_dir, "Master_Performance_Summary_Table.png")
save_dataframe_as_image(master_df, master_img_path, title="Master Performance Summary (All 10 Runs)")

print(f"\nSuccessfully generated charts, CSV tables, and Surf Green PNG image tables!")
print(f"Master summary image saved to: {master_img_path}")