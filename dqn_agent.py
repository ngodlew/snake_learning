import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from other import Direction, SnakeEnv
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = 64

        self.model = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=20000)

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_dim)  # Explore: select a random action
        state_t = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()  # Exploit: select the action with max Q-value

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor(np.array([t[1] for t in batch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor(np.array([t[4] for t in batch]))

        q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_agent(self):
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'memory_len': len(self.memory)
        }

        with open('dqn_agent.pkl', 'wb') as f:
            pickle.dump(save_data, f)

if __name__ == "__main__":
    MOVE_LEFT = 0
    MOVE_STRAIGHT = 1
    MOVE_RIGHT = 2

    file = open("dqnlearn-stats.csv", "a")
    file.write("Episode,Score,Total Steps,Death,Epsilon\n")
    file.flush()

    env = SnakeEnv(render_mode="human")
    dqn_agent = DQNAgent(state_dim=env.observation_space.n, action_dim=env.action_space.n)
    print("Starting DQN training...")

    for episode in range(10000):
        if episode % 10 == 0:
            print("Episode:", episode, " - Epsilon:", dqn_agent.epsilon)
        state, _ = env.reset()
        done = False

        while not done:
            move = dqn_agent.get_action(state)

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
            done = terminated or truncated

            dqn_agent.store_transition(state, move, reward, next_state, done)
            dqn_agent.train()

            state = next_state

            if done:
                stats = env._get_stats()
                file.write(f"{episode + 1}, {stats[0]}, {stats[1]}, {stats[2]}, {dqn_agent.epsilon}\n")
                file.flush()
                dqn_agent.epsilon = max(dqn_agent.epsilon_end, dqn_agent.epsilon * dqn_agent.epsilon_decay)

    print("Training Complete!\n")
    print("Saving dqn values...\n")
    dqn_agent.save_agent()
    print("DQN values saved to dqn_agent.pkl\n")