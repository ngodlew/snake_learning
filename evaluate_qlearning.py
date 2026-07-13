"""
Evaluate a trained Q-learning Snake agent.

Loads a previously trained Q-table and evaluates
the agent without further learning.

During evaluation:
    - Exploration is disabled.
    - The agent always selects the highest-value action.
    - Performance is measured using score and survival time.
"""

# Import pickle to load the trained Q-table
import pickle

# Import our QLearningWrapper class that wraps the Snake environment for use with GridMind
from qlearning_wrapper import QLearningWrapper

# Import the QLearning algorithm
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning

# Create environment
env = QLearningWrapper()

# Create Q-learning agent
agent = QLearning(
    env=env,
    summary_dir="./logs/evaluation"
)

# Load trained Q-table
with open("q_table.pkl", "rb") as file:
    q_table = pickle.load(file)

# Replace the agent's empty Q-table with the trained table.
# Both the learning algorithm and policy need to reference
# the same Q-table for correct action selection.
agent.q_values = q_table
agent.policy.Q = q_table

# Disable exploration (always select the best action)
agent.epsilon = 0
agent.policy.epsilon = 0

# Evaluation settings
num_episodes = 100

scores = []
steps_survived = []

# Run evaluation episodes
for episode in range(num_episodes):
    # Reset environment for new episode
    state, info = env.reset()

    total_steps = 0
    terminated = False
    truncated = False

    # Continue until the episode ends (snake dies or time limit reached)
    while not terminated and not truncated:
        # Select best action using learned Q-values
        action = agent.policy.get_action(state)

        # Take action
        state, reward, terminated, truncated, info = env.step(action)

        total_steps += 1
        print(
        "State:",
        state,
        "Action:",
        action
    )

    # Store results
    scores.append(env.env.score)
    steps_survived.append(total_steps)


    print(
        "Episode:",
        episode + 1,
        "Score:",
        env.env.score,
        "Steps:",
        total_steps
    )

# Print summary statistics
print("\nEvaluation Complete")
print("-------------------")
print("Average Score:", sum(scores) / len(scores))
print("Best Score:", max(scores))
print("Average Steps Survived:", sum(steps_survived) / len(steps_survived))