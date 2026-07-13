"""
Wrapper that adapts the Snake Gymnasium environment for GridMind Q-Learning.

The original SnakeEnv returns NumPy arrays as observations.
GridMind's tabular Q-learning stores Q-values in a dictionary, so states
must be hashable. This wrapper converts state arrays into tuples.
"""

# Import necessary libraries from Gymnamisum
import gymnasium as gym

# Import our implementation of the Gymnasium Snake environment
from environment import SnakeEnv

# Wrapper class for compatibility between the Snake Gymnasium environment and GridMind's Q-learning algorithm
class QLearningWrapper(gym.Env):
    def __init__(self):

        super().__init__()

        # Create an instance of the Snake Gymnasium environment
        self.env = SnakeEnv()

        # The available actions are identical to the Snake environment:
        # 0 = straight, 1 = left, 2 = right        
        self.action_space = self.env.action_space

        # Not required by GridMind, but defined for compatibility with Gymnasium
        self.observation_space = None

        # GridMind expects a Gymnasium environment specification        
        self.spec = self.env.spec

    # Reset the environment and return the initial state as a tuple
    def reset(self, seed=None, options=None):
        state, info = self.env.reset(
            seed=seed,
            options=options
        )

        return tuple(state), info

    # Take a step in the environment and return the next state as a tuple
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        return (
            tuple(state), # State as a tuple for hashability
            reward,       # Reward from the environment
            terminated,   # Whether the episode has ended
            truncated,    # Whether the episode was truncated due to time limits
            info          # Additional information from the environment
        )