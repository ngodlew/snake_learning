"""
Train a Deep Q-Network (DQN) reinforcement learning agent for the Snake game.

Uses Stable-Baselines3 to train a neural network-based agent
that learns to play Snake by interacting with the Gymnasium environment.

Unlike tabular Q-learning, DQN uses a neural network to approximate Q-values,
allowing it to generalize across similar states instead of storing each state
individually in a Q-table.

The trained model is saved as "snake_dqn" and can be loaded later for evaluation.
"""

# Import for the DQN algorithm from Stable Baselines 3
from stable_baselines3 import DQN

# Import our implementation of the Gymnasium Snake environment
from environment import SnakeEnv

# Create environment
env = SnakeEnv()

# Create DQN agent
model = DQN(
    "MlpPolicy",               # Use a neural network policy for function approximation
    env,                       # Use the Snake environment
    verbose=1,                 # Enable verbose output for training progress
    learning_rate=0.001,       # Controls how quickly the neural network updates its weights based on the error between predicted and target Q-values
    buffer_size=50000,         # Set the size of the replay buffer
    learning_starts=1000,      # Start learning after 1000 timesteps
    batch_size=64,             # Set the batch size for training (number of samples from the replay buffer)
    gamma=0.9,                 # Set the discount factor for future rewards
    exploration_fraction=0.2,  # Set the fraction of total timesteps for exploration decay
    exploration_final_eps=0.05 # Set the final exploration rate after decay
)

# Train model
model.learn(
    total_timesteps=500000
)

# Save trained model
model.save("snake_dqn")

print("Training complete!")