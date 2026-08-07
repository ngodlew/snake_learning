# snake_learning
Snake AI Learning

## Proposal
[Link](https://pennstateoffice365-my.sharepoint.com/:w:/r/personal/cjs8720_psu_edu/_layouts/15/Doc.aspx?sourcedoc=%7BA8D499A2-D3EA-4638-8F9A-43B52A4AA5F9%7D&file=AI801%20-%20Group%206%20Project%20Proposal.docx&fromShare=true&action=default&mobileredirect=true)
___
## Summary
A reinforcement learning implementation of the classic Snake game using **Gymnasium**, **GridMind Q-Learning**, and **Stable Baselines3 Deep Q-Networks (DQN)**.

This project converts a traditional Snake game into a reinforcement learning environment and compares multiple agents:

- Random Agent (baseline)
- Tabular Q-Learning Agent
- Deep Q-Network (DQN) Agent

---

## Environment

The Snake game is converted into a Gymnasium-compatible environment.

## Project Structure

snake_learning

│
├── dqn_agent.py
│   └── RL agent to play Snake using DQN within the snake environment
│
├── human.py
│   └── Human plugin to the snake environment for user input to play Snake
│
├── q_learning_agent.py
│   └── RL agent to play Snake using QLearning within the snake environment
│
├── snake_environment.py
│   └── Base Snake game environment as well as added features to support RL
│
└── environment.yml
    └── Script to allow for proper packages to be loaded within conda (Optional).
        This will also install libraries required for the Archived project.

## External Dependencies

This project requires the following external (non-standard) libraries:

- `gymnasium` - Provides the reinforcement learning environment interface (Version 1.0.0)
- `numpy` - Used for numerical operations and state representation
- `torch` - Deep learning framework used by Stable Baselines 3
- `pygame` - Provides support for the visual and input elements of Snake

## Commands

**To launch a human playable game of Snake**

`python human.py`

**To start training a QLearning agent**

`python q_learning_agent.py`

**To start training a DQN agent**

`python dqn_agent.py`

**NOTE:** The training of the DQN and the QLearning agents will, if run to completion of the
      program, will produce a .csv file containing the statistics of the run as well as a
      .pkl file of the relevant training data to reproduce the agent.

## Archived

### Old Project Structure

```
snake_learning/archive

│
├── environment.py
│   └── Gymnasium Snake environment
│
├── snake.py
│   └── Original Snake game implementation
│
├── qlearning_wrapper.py
│   └── Adapter that converts Snake states for GridMind Q-Learning
│
├── train_qlearning.py
│   └── Trains the tabular Q-Learning agent
│
├── evaluate_qlearning.py
│   └── Evaluates the trained Q-Learning agent
│
├── train_dqn.py
│   └── Trains the DQN agent
│
├── evaluate_dqn.py
│   └── Evaluates the trained DQN agent
│
├── random_agent.py
│   └── Random policy baseline
│
└── main.py
    └── Project menu interface
```

### Dependencies

This project requires the following libraries:

- `gymnasium` - Provides the reinforcement learning environment interface
- `numpy` - Used for numerical operations and state representation
- `stable-baselines3` - Provides the Deep Q-Network (DQN) implementation
- `torch` - Deep learning framework used by Stable Baselines 3
- `gridmind` - Provides the tabular Q-learning implementation
- `tqdm` - Displays training progress bars
- `tensorboard` - Used for logging and monitoring training runs

