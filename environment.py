
"""
Gymnasium environment for the Snake reinforcement learning project.

This class converts the existing Snake game into a Markov Decision Process
that can be used by reinforcement learning algorithms.

Actions:
    0 - Continue straight
    1 - Turn left
    2 - Turn right

State representation:
    The agent receives an 11-feature vector containing:
        - Collision danger information
        - Current snake direction
        - Relative food location
"""

# Import necessary libraries from Gymnamisum
import gymnasium as gym
from gymnasium import spaces

# Import numpy for state vector representation
import numpy as np

# Import our implementation of the Snake game
import snake

# Gynmasium environment for the Snake game
class SnakeEnv(gym.Env):
    def __init__(self):
        # Initialize the Snake environment
        super().__init__()

        # Three possible actions:
        # 0 = continue forward
        # 1 = turn left
        # 2 = turn right
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(11,),
            dtype=np.float32
        )

        self.score = 0

    # Reset the environment to its initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        snake.reset_grid()

        self.snake_list = [
            [5,5],
            [5,4],
            [5,3],
            [5,2]
        ]

        # Set initial direction of the snake to UP
        self.direction = snake.Direction.UP

        self.food_pos = [6,5]

        self.score = 0
        self.growth = 0

        state = self.get_state()

        return state, {}
    
    # Move the snake in the current direction
    def move_snake(self):
        match self.direction:
            case snake.Direction.LEFT:
                self.snake_list.insert(0, [self.snake_list[0][0] - 1, self.snake_list[0][1]])

            case snake.Direction.RIGHT:
                self.snake_list.insert(0, [self.snake_list[0][0] + 1, self.snake_list[0][1]])

            case snake.Direction.UP:
                self.snake_list.insert(0, [self.snake_list[0][0], self.snake_list[0][1] + 1])

            case snake.Direction.DOWN:
                self.snake_list.insert(0, [self.snake_list[0][0], self.snake_list[0][1] - 1])

        # Remove tail unless the snake is growing
        if self.growth > 0:
            self.growth -= 1
        else:
            self.snake_list.pop()
    
    # Change the direction of the snake based on the action taken by the agent
    def change_direction(self, action):
        current = self.direction

        # 0 = continue straight
        if action == 0:
            return

        # 1 = turn left
        elif action == 1:
            if current == snake.Direction.UP:
                self.direction = snake.Direction.LEFT

            elif current == snake.Direction.LEFT:
                self.direction = snake.Direction.DOWN

            elif current == snake.Direction.DOWN:
                self.direction = snake.Direction.RIGHT

            elif current == snake.Direction.RIGHT:
                self.direction = snake.Direction.UP


        # 2 = turn right
        elif action == 2:
            if current == snake.Direction.UP:
                self.direction = snake.Direction.RIGHT

            elif current == snake.Direction.RIGHT:
                self.direction = snake.Direction.DOWN

            elif current == snake.Direction.DOWN:
                self.direction = snake.Direction.LEFT

            elif current == snake.Direction.LEFT:
                self.direction = snake.Direction.UP

    # Check for collisions with walls or self
    def check_collision(self):
        head = self.snake_list[0]

        # Wall collision
        if (head[0] == 0 or head[0] == snake.GRID_COL - 1 or head[1] == 0 or head[1] == snake.GRID_ROW - 1):
            return True

        # Self collision
        for body in self.snake_list[1:]:
            if head == body:
                return True

        return False

    # Advance the environment by one step based on the action taken by the agent
    def step(self, action):
        # Change snake direction based on agent action
        self.change_direction(action)

        # Move snake one step forward
        self.move_snake()

        # Default reward for surviving one move
        reward = -0.1

        # Check if snake ate food
        if self.snake_list[0] == self.food_pos:
            # Increase score
            self.score += 1

            # Grow snake on the next movement
            self.growth += snake.GROWTH_LENGTH

            # Reward for eating food
            reward = 10

            # Place new food
            self.food_pos = snake.place_food()


        # Update the game grid to reflect the new snake position
        snake.game_grid = [
            [snake.Status.NONE for _ in range(snake.GRID_COL)]
            for _ in range(snake.GRID_ROW)
        ]

        for segment in self.snake_list:
            snake.game_grid[segment[0]][segment[1]] = snake.Status.SNAKE

        # Check if snake collided with wall or itself
        terminated = self.check_collision()

        if terminated:
            reward = -10

        # Currently there is no time limit for the game
        truncated = False

        # Get the new state after taking the action
        state = self.get_state()

        return (
            state,      # New observation/state after action
            reward,     # Value returned from environment
            terminated, # True if episode ended due to game over
            truncated,  # True if episode ended due to time limit
            {}
        )

    # Get the current state of the environment as a feature vector
    def get_state(self):
        head = self.snake_list[0]

        # Check positions around the snake head
        danger_straight = self.is_collision(
            self.get_next_position(self.direction)
        )

        danger_left = self.is_collision(
            self.get_next_position(self.get_left_direction())
        )

        danger_right = self.is_collision(
            self.get_next_position(self.get_right_direction())
        )

        # Current direction
        direction_up = self.direction == snake.Direction.UP
        direction_down = self.direction == snake.Direction.DOWN
        direction_left = self.direction == snake.Direction.LEFT
        direction_right = self.direction == snake.Direction.RIGHT


        # Food location relative to snake
        food_left = self.food_pos[0] < head[0]
        food_right = self.food_pos[0] > head[0]
        food_up = self.food_pos[1] > head[1]
        food_down = self.food_pos[1] < head[1]

        # Create the state vector as a numpy array
        state = np.array([
            danger_straight,
            danger_left,
            danger_right,

            direction_up,
            direction_down,
            direction_left,
            direction_right,

            food_left,
            food_right,
            food_up,
            food_down,
        ], dtype=np.float32)

        return state

    # Get the next position of the snake head based on the current direction
    def get_next_position(self, direction):
        head = self.snake_list[0]

        if direction == snake.Direction.UP:
            return [head[0], head[1] + 1]
        elif direction == snake.Direction.DOWN:
            return [head[0], head[1] - 1]
        elif direction == snake.Direction.LEFT:
            return [head[0] - 1, head[1]]
        elif direction == snake.Direction.RIGHT:
            return [head[0] + 1, head[1]]
    
    # Get the direction to the right of the current direction
    def get_right_direction(self):
        if self.direction == snake.Direction.UP:
            return snake.Direction.RIGHT
        elif self.direction == snake.Direction.RIGHT:
            return snake.Direction.DOWN
        elif self.direction == snake.Direction.DOWN:
            return snake.Direction.LEFT
        else:
            return snake.Direction.UP

    # Get the direction to the left of the current direction
    def get_left_direction(self):
        if self.direction == snake.Direction.UP:
            return snake.Direction.LEFT
        elif self.direction == snake.Direction.LEFT:
            return snake.Direction.DOWN
        elif self.direction == snake.Direction.DOWN:
            return snake.Direction.RIGHT
        else:
            return snake.Direction.UP
        
    # Check if the given position would result in a collision
    def is_collision(self, position):
        # Wall collision
        if (position[0] == 0 or position[0] == snake.GRID_COL - 1 or position[1] == 0 or position[1] == snake.GRID_ROW - 1):
            return True

        # Snake body collision
        if position in self.snake_list:
            return True

        return False