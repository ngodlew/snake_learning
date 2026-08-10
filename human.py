import sys
import pygame
from snake_environment import SnakeEnv, Direction

def main():
    env = SnakeEnv(render_mode="human", render_fps=5)
    state, info = env.reset()
    previous_action = Direction.UP  # Default initial action    

    print("--- Manual Control ---")

    running = True
    while running:
        action = previous_action  # Default to the previous action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and previous_action != Direction.DOWN:
                    action = Direction.UP  # UP
                elif event.key == pygame.K_DOWN and previous_action != Direction.UP:
                    action = Direction.DOWN  # DOWN
                elif event.key == pygame.K_LEFT and previous_action != Direction.RIGHT:
                    action = Direction.LEFT  # LEFT
                elif event.key == pygame.K_RIGHT and previous_action != Direction.LEFT:
                    action = Direction.RIGHT  # RIGHT

        previous_action = action
        state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Game Over! Resetting the environment.")
            state, info = env.reset()
            previous_action = Direction.UP  # Reset to default action after game over

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()