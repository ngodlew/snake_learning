import csv
import os
import pandas as pd
from snake_environment import SnakeEnv, Direction

# Set up directory for output CSV files
BASE_OUTPUT_DIR = r"C:\Users\rockl\PycharmProjects\SnakeGame\snake_learning-main\RandomAgentResults"

# Explicit target folder paths for the 5 runs
RUN_FOLDERS = [
    os.path.join(BASE_OUTPUT_DIR, "1"),
    os.path.join(BASE_OUTPUT_DIR, "2"),
    os.path.join(BASE_OUTPUT_DIR, "3"),
    os.path.join(BASE_OUTPUT_DIR, "4"),
    os.path.join(BASE_OUTPUT_DIR, "5"),
]

# Action mapping: 0 = LEFT, 1 = STRAIGHT, 2 = RIGHT
MOVE_LEFT = 0
MOVE_STRAIGHT = 1
MOVE_RIGHT = 2


def convert_relative_to_direction(current_direction, move):
    """Maps relative moves (Left, Straight, Right) to the Directions."""
    action = current_direction
    match current_direction:
        case Direction.UP:
            if move == MOVE_LEFT:
                action = Direction.LEFT
            elif move == MOVE_RIGHT:
                action = Direction.RIGHT
        case Direction.DOWN:
            if move == MOVE_LEFT:
                action = Direction.RIGHT
            elif move == MOVE_RIGHT:
                action = Direction.LEFT
        case Direction.LEFT:
            if move == MOVE_LEFT:
                action = Direction.DOWN
            elif move == MOVE_RIGHT:
                action = Direction.UP
        case Direction.RIGHT:
            if move == MOVE_LEFT:
                action = Direction.UP
            elif move == MOVE_RIGHT:
                action = Direction.DOWN
    return action


def run_simulation(output_dir, run_index):
    # Ensure the subfolder exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the Snake environment
    env = SnakeEnv()
    episodes = 10000

    scores = []
    steps = []

    # File paths
    stats_path = os.path.join(output_dir, "random_agent_stats.csv")
    dataset_path = os.path.join(output_dir, "random_agent_dataset.csv")

    # File 1: Performance statistics
    stats_file = open(stats_path, "w", newline="")
    stats_writer = csv.writer(stats_file)
    stats_writer.writerow(["Episode", "Score", "Total Steps", "Death"])

    # File 2: Feature & target dataset
    dataset_file = open(dataset_path, "w", newline="")
    dataset_writer = csv.writer(dataset_file)

    obs_dim = env.observation_space.shape[0]
    feature_cols = [f"obs_{i}" for i in range(obs_dim)] + ["action"]
    target_cols = ["reward"] + [f"next_obs_{i}" for i in range(obs_dim)] + ["done"]

    # Write header and row
    dataset_writer.writerow(feature_cols + target_cols)

    print(f"\n=================== Starting Run {run_index} ===================")
    print(f"Directory: {output_dir}")

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_steps = 0

        while not terminated and not truncated:
            # 1. Sample a random relative move (0: Left, 1: Straight, 2: Right)
            move = env.action_space.sample()

            # 2. Derive absolute Direction from current orientation
            current_direction = Direction(
                state[0] * 0 + state[1] * 1 + state[2] * 2 + state[3] * 3
            )
            action = convert_relative_to_direction(current_direction, move)

            # 3. Step through the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            # 4. Save step-level Features & Targets
            features = list(state) + [move]
            targets = [reward] + list(next_state) + [int(done)]
            dataset_writer.writerow(features + targets)

            state = next_state

        # Log episode metrics
        scores.append(env.score)
        steps.append(total_steps)
        death_type = env.death.name if hasattr(env, "death") else "NONE"
        stats_writer.writerow([episode + 1, env.score, total_steps, death_type])

        if (episode + 1) % 2000 == 0:
            print(f"Run {run_index} | Episode: {episode + 1} | Avg Score (of last 1000 EP): {sum(scores[-1000:]) / 1000:.2f}")

    stats_file.close()
    dataset_file.close()

    print(f"Run {run_index} Complete | Avg Score: {sum(scores) / len(scores):.2f} | Best Score: {max(scores)}")


def main():
    for idx, folder in enumerate(RUN_FOLDERS, start=1):
        run_simulation(folder, idx)

    print("\nAll 5 Runs Saved Successfully!")


if __name__ == "__main__":
    main()