"""
Evaluate a trained Deep Q-Network (DQN) reinforcement learning agent.

Loads a previously trained DQN model and evaluates its performance
on the Snake Gymnasium environment. The agent uses the learned neural network
to select actions without exploration and attempts to maximize its score.

The evaluation runs multiple episodes and calculates performance metrics:
    - Average score achieved
    - Best score achieved
    - Average number of steps survived
"""

# Import the DQN algorithm from Stable Baselines 3
from stable_baselines3 import DQN

# Import our implementation of the Gymnasium Snake environment
from environment import SnakeEnv

# Create the Snake environment
env = SnakeEnv()

# Load the trained DQN model from the file "snake_dqn.zip" and associate it with the Snake environment
model = DQN.load(
    "snake_dqn",
    env=env
)

# Set the number of evaluation episodes to run
episodes = 100

scores = []
steps = []

# Run evaluation episodes
for episode in range(episodes):
    state, info = env.reset()

    terminated = False
    truncated = False

    total_steps = 0

    # Continue until the episode ends (snake dies or time limit reached)
    while not terminated and not truncated:
        action, _ = model.predict(
            state,
            deterministic=True
        )

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


print("\nEvaluation Complete")
print("-------------------")
print("Average Score:", sum(scores)/len(scores))
print("Best Score:", max(scores))
print("Average Steps:", sum(steps)/len(steps))