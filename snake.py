# Imports
import pygame   # Need to include
import time
import random
from enum import Enum

# Enumerations
class Direction(Enum):
    UP    = 1
    DOWN  = 2
    LEFT  = 3
    RIGHT = 4
    
class Status(Enum):
    NONE  = 1
    SNAKE = 2
    WALL  = 3
    FOOD  = 4

# Global Declarations/Constants\
GRID_COL = 12
GRID_ROW = 12
SNAKE_SIZE  = 20
SNAKE_SPEED = 5
SCORE_HEIGHT = 40
FONT_SIZE = 25
GROWTH_LENGTH = 1

SCREEN_HEIGHT = GRID_ROW * SNAKE_SIZE + SCORE_HEIGHT
SCREEN_WIDTH  = GRID_COL * SNAKE_SIZE

white = (255, 255, 255)
black = (0, 0, 0)
red =   (255, 0, 0)
green = (0, 255, 0)
blue =  (50, 150, 215)

# Initial Setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()
font_style = pygame.font.SysFont("bahnschrift", FONT_SIZE)

game_grid = [[Status.NONE for _ in range(GRID_COL)] for _ in range(GRID_ROW)]

for col in range(GRID_COL):
    game_grid[col][0] = Status.WALL
    game_grid[col][GRID_ROW - 1] = Status.WALL
    
for row in range(GRID_ROW):
    game_grid[0][row] = Status.WALL
    game_grid[GRID_COL - 1][row] = Status.WALL

def display_text(score):
    value = font_style.render(f"Score: {score}", True, white)
    screen.blit(value, [4, 8])
    
def coordinates(x, y):
    return [x * SNAKE_SIZE, (GRID_ROW - y - 1) * SNAKE_SIZE + SCORE_HEIGHT, SNAKE_SIZE, SNAKE_SIZE]
    
def draw_snake_head(head_pos, direction):
    head_coords = coordinates(head_pos[0], head_pos[1])

    match direction:
        case Direction.UP:
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
        case Direction.DOWN:
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
        case Direction.LEFT:
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.2, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
        case Direction.RIGHT:
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            pygame.draw.rect(screen, red, [head_coords[0] + SNAKE_SIZE * 0.6, head_coords[1] + SNAKE_SIZE * 0.6, SNAKE_SIZE * 0.2, SNAKE_SIZE * 0.2])
            
def place_food():
    blank_spots = []
    
    for col in range(1, GRID_COL - 1):
        for row in range(1, GRID_ROW - 1):
            if game_grid[col][row] == Status.NONE:
                blank_spots.append([col, row])

    loc = random.randrange(0, len(blank_spots) - 1)
    game_grid[blank_spots[loc][0]][blank_spots[loc][1]] = Status.FOOD
    print([blank_spots[loc][0]], [blank_spots[loc][1]])
    return [blank_spots[loc][0],blank_spots[loc][1]]
    
def reset_grid():
    for col in range(GRID_COL):
        game_grid[col][0] = Status.WALL
        game_grid[col][GRID_ROW - 1] = Status.WALL
        
    for row in range(GRID_ROW):
        game_grid[0][row] = Status.WALL
        game_grid[GRID_COL - 1][row] = Status.WALL
        
    for col in range (1, GRID_COL - 1):
        for row in range (1, GRID_ROW - 1):
            game_grid[col][row] = Status.NONE

def game_loop():
    game_over  = False
    game_close = False
    
    food_eaten = 0
    snake_length = 4
    growth = 0
    
    reset_grid()
    
    snake_list = [[5,5],[5,4],[5,3],[5,2]]
    
    game_grid[5][5] = Status.SNAKE
    game_grid[5][4] = Status.SNAKE
    game_grid[5][3] = Status.SNAKE
    game_grid[5][2] = Status.SNAKE
    
    food_pos = place_food()
    
    current_direction = Direction.UP
    direction = Direction.UP
    
    while not game_over:
        while game_close:
            screen.fill(black)
            message1 = font_style.render("Game Over!", True, red)
            message2 = font_style.render("Q - Quit", True, red)
            message3 = font_style.render("C - Restart", True, red)
            screen.blit(message1, [SCREEN_WIDTH / 6, SCREEN_HEIGHT / 4])
            screen.blit(message2, [SCREEN_WIDTH / 6 + FONT_SIZE, SCREEN_HEIGHT / 3 + FONT_SIZE])
            screen.blit(message3, [SCREEN_WIDTH / 6 + FONT_SIZE, SCREEN_HEIGHT / 3 + 2 * FONT_SIZE])
            display_text(food_eaten)
            pygame.display.update()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over  = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and current_direction != Direction.RIGHT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and current_direction != Direction.LEFT:
                    direction = Direction.RIGHT
                elif event.key == pygame.K_UP and current_direction != Direction.DOWN:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN and current_direction != Direction.UP:
                    direction = Direction.DOWN
        
        screen.fill(black)
        
        current_direction = direction
        
        # Move and grow the snake
        match direction:
            case Direction.LEFT:
                snake_list.insert(0, [snake_list[0][0] - 1, snake_list[0][1]])
            case Direction.RIGHT:
                snake_list.insert(0, [snake_list[0][0] + 1, snake_list[0][1]])
            case Direction.UP:
                snake_list.insert(0, [snake_list[0][0], snake_list[0][1] + 1])
            case Direction.DOWN:
                snake_list.insert(0, [snake_list[0][0], snake_list[0][1] - 1])
                
        game_grid[snake_list[0][0]][snake_list[0][1]] = Status.SNAKE
        
        if growth > 0:
            growth = growth - 1
        else:
            game_grid[snake_list[-1][0]][snake_list[-1][1]] = Status.NONE
            snake_list.pop()
        
        # Collision checks
        if snake_list[0] == food_pos:
            food_eaten = food_eaten + 1
            growth = growth + GROWTH_LENGTH
            food_pos = place_food()
        elif snake_list[0][0] == 0 or snake_list[0][0] == GRID_COL - 1 or snake_list[0][1] == 0 or snake_list[0][1] == GRID_ROW - 1:
            game_close = True
        else:
            for body in snake_list[1:-1]:
                if snake_list[0] == body:
                    game_close = True
        
        for col in range(GRID_COL):
            for row in range(GRID_ROW):
                match game_grid[col][row]:
                    case Status.WALL:
                        pygame.draw.rect(screen, blue, coordinates(col, row))
                    case Status.SNAKE:
                        pygame.draw.rect(screen, green, coordinates(col, row))
                    case Status.FOOD:
                        pygame.draw.rect(screen, red, coordinates(col, row))
                        
        draw_snake_head(snake_list[0], direction)
        
        display_text(food_eaten)
                        
        pygame.display.update()
        
        clock.tick(SNAKE_SPEED)
        
    pygame.quit()
    quit()
    
game_loop()
    