"""
Evaluate a random agent on the Snake reinforcement learning environment.

Provides a baseline performance measurement by selecting actions
randomly without using any reinforcement learning algorithm.

The random agent is used to compare performance against learned agents such as
Q-learning and Deep Q-Networks (DQN).

Evaluation metrics:
    - Average score achieved
    - Best score achieved
    - Average number of steps survived
"""

# Import our Gymnasium Snake environment
from environment import SnakeEnv

# Create the Snake environment
env = SnakeEnv()

# Number of episodes to evaluate
episodes = 100

scores = []
steps = []

# Run evaluation episodes
for episode in range(episodes):

    state, info = env.reset()

    terminated = False
    truncated = False

    total_steps = 0


    # Continue until the snake dies
    while not terminated and not truncated:

        # Select a random action
        action = env.action_space.sample()

        # Take the action in the environment
        state, reward, terminated, truncated, info = env.step(action)

        total_steps += 1


    scores.append(env.score)
    steps.append(total_steps)


    print(
        "Episode:",
        episode + 1,
        "Score:",
        env.score,
        "Steps:",
        total_steps
    )

# Print summary statistics
print("\nRandom Agent Evaluation Complete")
print("-------------------------------")
print("Average Score:", sum(scores) / len(scores))
print("Best Score:", max(scores))
print("Average Steps:", sum(steps) / len(steps))