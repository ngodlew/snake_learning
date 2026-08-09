import numpy as np
import random
from snake_environment import SnakeEnv, Direction
# from other import SnakeEnv, Direction
import pickle

alpha = 0.1
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
gamma = 0.9

env = SnakeEnv(render_mode="human")
state_size = env.observation_space.n
action_size = env.action_space.n

MOVE_LEFT = 0
MOVE_STRAIGHT = 1
MOVE_RIGHT = 2

q_table = {}

file = open("qlearn-stats.csv", "a")
file.write("Episode,Score,Total Steps,Death,Epsilon\n")
file.flush()

for episode in range(10000):
    if episode % 100 == 0:
        print("Episode:", episode, " - Epsilon:", epsilon)
    state, _ = env.reset()
    done = False

    while not done:
        state_key = tuple(state)
        
        if random.uniform(0, 1) < epsilon:
            move = env.action_space.sample()
        else:
            q_values = [q_table.get((state_key, val), 0.0) for val in range(action_size)]
            move = int(np.argmax(q_values))

        current_direction = Direction(state[0] * 0 + state[1] * 1 + state[2] * 2 + state[3] * 3)

        action = Direction(current_direction)

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

        next_state, reward, terminated, truncated, _ = env.step(action)
        env._render_frame()
        done = terminated or truncated

        if not done:
            next_state_key = tuple(next_state)
            current_q_value = q_table.get((state_key, move), 0.0)
            next_q_values = [q_table.get((next_state_key, val), 0.0) for val in range(action_size)]
            max_next_q = max(next_q_values)

            new_q_value = current_q_value + alpha * (reward + (gamma * max_next_q) - current_q_value)
            q_table[(state_key, move)] = new_q_value
            state = next_state
        else:
            stats = env._get_stats()
            file.write(f"{episode + 1}, {stats[0]}, {stats[1]}, {stats[2]}, {epsilon}\n")
            file.flush()
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

file.close()

# Save the trained Q-table to a file for later evaluation
print("Training Complete!\n")
print("Saving q_table...\n")
with open("q_table.pkl", "wb") as file:
    pickle.dump(q_table, file)
print("q_table saved!\n")
