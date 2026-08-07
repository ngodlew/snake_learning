"""
Main menu for the Snake reinforcement learning project.

Provides a simple interface for running:
    - Manual Snake gameplay
    - Q-Learning training and evaluation
    - DQN training and evaluation
    - Random agent baseline evaluation
"""

import subprocess
import sys

# Display the main menu options
def display_menu():
    print("\n================================")
    print(" Snake Reinforcement Learning")
    print("================================")
    print("1. Play Snake manually")
    print("2. Train Q-Learning agent")
    print("3. Evaluate Q-Learning agent")
    print("4. Train DQN agent")
    print("5. Evaluate DQN agent")
    print("6. Run Random Baseline")
    print("7. Exit")
    print("================================")

# Run a Python script as a subprocess
def run_script(script_name):
    process = subprocess.Popen([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    std_out, std_err = process.communicate()

    if std_err:
        print(f"Error running {script_name}:\n{std_err}")
    else:
        print(f"Output from {script_name}:\n{std_out}")


# Main function to handle user input and execute the selected option
def main():
    while True:
        display_menu()

        choice = input("Select option: ")

        if choice == "1":
            print("Starting manual Snake...")
            run_script("snake.py")
        elif choice == "2":
            print("Training Q-Learning agent...")
            run_script("train_qlearning.py")
        elif choice == "3":
            print("Evaluating Q-Learning agent...")
            run_script("evaluate_qlearning.py")
        elif choice == "4":
            print("Training DQN agent...")
            run_script("train_dqn.py")
        elif choice == "5":
            print("Evaluating DQN agent...")
            run_script("evaluate_dqn.py")
        elif choice == "6":
            print("Running random agent baseline...")
            run_script("random_agent.py")
        elif choice == "7":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please select 1-7.")

if __name__ == "__main__":
    main()