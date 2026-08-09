from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

GRID_COL = 10
GRID_ROW = 10
SNAKE_SIZE  = 20
SNAKE_SPEED = 5
SCORE_HEIGHT = 40
FONT_SIZE = 25
GROWTH_LENGTH = 1

white = (255, 255, 255)
black = (0, 0, 0)
red =   (255, 0, 0)
green = (0, 255, 0)
blue =  (50, 150, 215)

class Status(Enum):
    NONE  = 1
    SNAKE = 2
    WALL  = 3
    FOOD  = 4

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, render_fps=1000):
        super().__init__()

        self.metadata = self.metadata.copy()
        self.metadata["render_fps"] = render_fps;

        self.game_grid = [[Status.NONE for _ in range(GRID_COL)] for _ in range(GRID_ROW)]

        # Define the action space for the environment
        self.action_space = spaces.Discrete(3)  # Left, Straight, Right

        # Define the observation space for the environment
        self.observation_space = spaces.MultiBinary(11)

        self.start_pos = (5, 5)  # Starting position of the snake
        self.agent_pos = self.start_pos  # Current position of the snake head
        self.body = []
        self.growth = 0
        self.hi_score = 0

        # Pygame UI Window Setup
        self.render_mode = render_mode
        self.grid_size = SNAKE_SIZE
        self.window_height = GRID_ROW * SNAKE_SIZE + SCORE_HEIGHT
        self.window_width = GRID_COL * SNAKE_SIZE
        self.window = None
        self.clock = None

    def _get_obs(self):

        # Direction
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT

        food_up = self.food_pos[1] < self.body[0][1]
        food_down = self.food_pos[1] > self.body[0][1]
        food_left = self.food_pos[0] < self.body[0][0]
        food_right = self.food_pos[0] > self.body[0][0]

        grid_up = Status.WALL
        grid_down = Status.WALL
        grid_left = Status.WALL
        grid_right = Status.WALL

        if self.body[0][1] > 0:
            grid_up = self.game_grid[self.body[0][0]][self.body[0][1] - 1]

        if self.body[0][1] < GRID_ROW - 2:
            grid_down = self.game_grid[self.body[0][0]][self.body[0][1] + 1]

        if self.body[0][0] > 0:
            grid_left = self.game_grid[self.body[0][0] - 1][self.body[0][1]]

        if self.body[0][0] < GRID_COL - 2:
            grid_right = self.game_grid[self.body[0][0] + 1][self.body[0][1]]
            # grid_left = self.game_grid[self.body[0][0] + 1][self.body[0][1]]

        danger_ahead = (dir_up and (grid_up == Status.WALL or grid_up == Status.SNAKE)) or \
                       (dir_down and (grid_down == Status.WALL or grid_down == Status.SNAKE)) or \
                       (dir_left and (grid_left == Status.WALL or grid_left == Status.SNAKE)) or \
                       (dir_right and (grid_right == Status.WALL or grid_right == Status.SNAKE))

        danger_left  = (dir_up and (grid_left == Status.WALL or grid_left == Status.SNAKE)) or \
                       (dir_down and (grid_right == Status.WALL or grid_right == Status.SNAKE)) or \
                       (dir_left and (grid_down == Status.WALL or grid_down == Status.SNAKE)) or \
                       (dir_right and (grid_up == Status.WALL or grid_up == Status.SNAKE))

        danger_right = (dir_up and (grid_right == Status.WALL or grid_right == Status.SNAKE)) or \
                       (dir_down and (grid_left == Status.WALL or grid_left == Status.SNAKE)) or \
                       (dir_left and (grid_up == Status.WALL or grid_up == Status.SNAKE)) or \
                       (dir_right and (grid_down == Status.WALL or grid_down == Status.SNAKE))

        obs_state = [dir_up,
                     dir_down,
                     dir_left,
                     dir_right,
                     food_up,
                     food_down,
                     food_left,
                     food_right,
                     danger_ahead,
                     danger_left,
                     danger_right]
        
        return np.array(obs_state, dtype=int)

    def _place_food(self):
        blank_cells = []
        # Place food in a random empty cell
        for col in range(1, GRID_COL - 1):
            for row in range(1, GRID_ROW - 1):
                if self.game_grid[col][row] == Status.NONE:
                    blank_cells.append([col, row])

        loc = random.randrange(0, len(blank_cells) - 1)
        self.game_grid[blank_cells[loc][0]][blank_cells[loc][1]] = Status.FOOD
        return [blank_cells[loc][0], blank_cells[loc][1]]

    def _get_stats(self):
        return [self.score, self.n_steps, self.death]

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        # Return the initial observation and an empty info dictionary
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.direction = Direction.UP  # Reset the direction to UP
        self.body = [[5, 5], [5, 4], [5, 3], [5, 2]]  # Reset the snake body to only the head
        self.n_steps = 0
        self.steps_since_food = 0
        self.time_penalty = 0
        self.death = Status.NONE

        for col in range(GRID_COL):
            self.game_grid[col][0] = Status.WALL
            self.game_grid[col][GRID_ROW - 1] = Status.WALL

        for row in range(GRID_ROW):
            self.game_grid[0][row] = Status.WALL
            self.game_grid[GRID_COL - 1][row] = Status.WALL

        for col in range(1, GRID_COL - 1):
            for row in range(1, GRID_ROW - 1):
                self.game_grid[col][row] = Status.NONE

        self.game_grid[5][5] = Status.SNAKE
        self.game_grid[5][4] = Status.SNAKE
        self.game_grid[5][3] = Status.SNAKE
        self.game_grid[5][2] = Status.SNAKE

        self.food_pos = self._place_food()

        self.score = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        row, col = self.agent_pos
        self.direction = action  # Update the direction based on the action taken

        match action:
            case Direction.UP:
                self.body.insert(0, [self.body[0][0], self.body[0][1] + 1])
            case Direction.DOWN:
                self.body.insert(0, [self.body[0][0], self.body[0][1] - 1])
            case Direction.LEFT:
                self.body.insert(0, [self.body[0][0] - 1, self.body[0][1]])
            case Direction.RIGHT:
                self.body.insert(0, [self.body[0][0] + 1, self.body[0][1]])

        self.game_grid[self.body[0][0]][self.body[0][1]] = Status.SNAKE
        self.agent_pos = self.body[0]

        if self.growth > 0:
            self.growth -= 1
        else:
            if self.body[0] != self.body[-1]:
                self.game_grid[self.body[-1][0]][self.body[-1][1]] = Status.NONE
            self.body.pop()

        if self.body[0] == self.food_pos:
            self.score += 1
            self.growth += GROWTH_LENGTH
            self.steps_since_food = 0
            self.food_pos = self._place_food()
        elif self.body[0][0] == 0 or self.body[0][0] == GRID_COL - 1 or self.body[0][1] == 0 or self.body[0][1] == GRID_ROW - 1:
            terminated = True
            self.death = Status.WALL
        else:
            for body in self.body[1:]:
                if self.agent_pos == body:
                    self.death = Status.SNAKE
                    terminated = True
                    break

        self.n_steps += 1

        if (self.steps_since_food > 20):
            self.time_penalty += 0.1

        if self.steps_since_food == 0:
            food_reward = 0
        else:
            food_reward = -1.5
            old_distance = abs(self.body[1][0] - self.food_pos[0]) + abs(self.body[1][1] - self.food_pos[1])
            new_distance = abs(self.body[0][0] - self.food_pos[0]) + abs(self.body[0][1] - self.food_pos[1])

            if new_distance < old_distance:
                food_reward = 1.0


        reward = 10 * self.score - 10 * terminated + food_reward # - self.time_penalty

        self.steps_since_food += 1

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def _draw_square(self, x, y):
        return [x * SNAKE_SIZE, (GRID_ROW - y - 1) * SNAKE_SIZE + SCORE_HEIGHT, SNAKE_SIZE, SNAKE_SIZE]

    def _draw_snake_head(self, canvas, head_pos, direction):
        head_coords = self._draw_square(head_pos[0], head_pos[1])

        match direction:
            case Direction.UP:
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            case Direction.DOWN:
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            case Direction.LEFT:
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            case Direction.RIGHT:
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
                pygame.draw.rect(canvas, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])

    def _display_text(self, canvas, score):
        value = pygame.font.SysFont("bahnschrift", FONT_SIZE).render(f"Score: {score}", True, white)
        canvas.blit(value, [4, 8])
        hs = pygame.font.SysFont("bahnschrift", FONT_SIZE).render(f"Hi: {self.hi_score}", True, white)
        canvas.blit(hs, [120, 8])

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Snake")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(black)

        for col in range(GRID_COL):
            for row in range(GRID_ROW):
                cell_status = self.game_grid[col][row]
                if cell_status == Status.WALL:
                    pygame.draw.rect(canvas, blue, self._draw_square(col, row))
                elif cell_status == Status.SNAKE:
                    pygame.draw.rect(canvas, green, self._draw_square(col, row))
                elif cell_status == Status.FOOD:
                    pygame.draw.rect(canvas, red, self._draw_square(col, row))

        if self.score > self.hi_score:
            self.hi_score = self.score

        self._draw_snake_head(canvas, self.body[0], self.direction)
        self._display_text(canvas, self.score)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"]) 

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
