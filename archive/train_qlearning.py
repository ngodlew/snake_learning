"""
Train a tabular Q-learning agent to play Snake.

1. Creates the Gymnasium Snake environment.
2. Creates a GridMind Q-learning agent.
3. Trains the agent through repeated episodes.
4. Saves the learned Q-table for later evaluation.

The trained Q-table is stored as q_table.pkl and can be loaded
by evaluate_qlearning.py.
"""

# Import our QLearningWrapper class that wraps the Snake environment for use with GridMind
from qlearning_wrapper import QLearningWrapper

# Import the QLearning algorithm from GridMind
from gridmind.algorithms.tabular.temporal_difference.control.q_learning import QLearning

# Import pickle to save the trained model
import pickle

# Create the Snake environment wrapped for Q-learning
env = QLearningWrapper()

# Create the Q-learning agent with the environment and specify a directory for logging (necessary for GridMind)
agent = QLearning(
    env=env,
    summary_dir="./logs/q_learning",
    epsilon=1.0
)

# Set the agent's Q-table to the agent's policy Q-table 
# so that they are the same object and updates to one will reflect in the other
agent.policy.Q = agent.q_values

# Set epsilon decay to True to enable exploration decay over time
agent.epsilon_decay = True

# Train the agent for a specified number of episodes
agent.optimize_policy(
    num_episodes=10000
)

# Save the trained Q-table to a file for later evaluation
with open("q_table.pkl", "wb") as file:
    pickle.dump(dict(agent.q_values), file)

print("Training complete!")