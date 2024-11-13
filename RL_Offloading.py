import numpy as np
import matplotlib.pyplot as plt

# Define the reinforcement learning agent
class RLAgent:
    def __init__(self, num_states, num_actions):
        """
        Initializing the Q-table with zeros for the specified number of states and actions.
        
        Parameters:
        - num_states: Number of possible states in the environment
        - num_actions: Number of possible actions in each state
        """
        self.q_table = np.zeros((num_states, num_actions))  # Initialize Q-table with zeros

    def choose_action(self, state):
        """
        Choose an action based on the highest Q-value for the given state.
        
        Parameters:
        - state: Current state of the agent
        
        Returns:
        - action with the highest Q-value
        """
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, learning_rate=0.1, discount_factor=0.9):
        """
        Update the Q-value for a specific state-action pair using the Q-learning formula.
        
        Parameters:
        - state: Current state
        - action: Action taken
        - reward: Reward received after taking the action
        - learning_rate: Learning rate for Q-value update (default 0.1)
        - discount_factor: Discount factor for future rewards (default 0.9)
        """
        max_future_q = np.max(self.q_table[state])  # Highest Q-value in the next state
        current_q = self.q_table[state, action]  # Current Q-value
        # Calculate the new Q-value
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        self.q_table[state, action] = new_q  # Update the Q-table with the new Q-value

# Main function to initialize and run the Q-learning agent
def main():
    try:
        # User input for number of states and actions in the environment
        num_states = int(input("Enter the number of states: "))
        num_actions = int(input("Enter the number of actions: "))

        # Initialize the RL agent with the specified states and actions
        agent = RLAgent(num_states, num_actions)
        
        # User input for the state to track Q-value progression
        track_state = int(input(f"Enter the state you want to track (0 to {num_states - 1}): "))

        # Set up the plot for real-time Q-value progression for the tracked state
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        # Create a line for each action in the tracked state
        lines = [ax.plot([], [], label=f"Action {a}")[0] for a in range(num_actions)]
        ax.set_xlim(0, 10)  # Initial x-axis limit
        ax.set_ylim(-1, 10)  # y-axis limit based on expected Q-value range
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Q-Value")
        ax.set_title(f"Q-Value Progression for State {track_state}")
        ax.legend()

        iteration = 0
        # Run the Q-learning update in a loop
        while True:
            # User input for state, action, and reward
            state = int(input(f"Enter the state (0 to {num_states - 1}): "))
            action = int(input(f"Enter the action (0 to {num_actions - 1}): "))
            reward = float(input("Enter the reward: "))

            # Agent chooses an action based on the current state
            chosen_action = agent.choose_action(state)
            print(f"Chosen action for state {state}: {chosen_action}")

            # Update the Q-value in the Q-table for the state-action pair
            agent.update_q_value(state, action, reward)

            # Display the updated Q-table after each iteration
            print("Updated Q-Table:")
            print(agent.q_table)

            # Update the plot for the tracked state with Q-values for each action
            for a in range(num_actions):
                # Append new data points for each action in the tracked state
                lines[a].set_xdata(np.append(lines[a].get_xdata(), iteration))
                lines[a].set_ydata(np.append(lines[a].get_ydata(), agent.q_table[track_state, a]))
            
            # Adjust x-axis limit if more iterations are added
            ax.set_xlim(0, max(10, iteration + 1))
            ax.relim()  # Recalculate limits based on new data
            ax.autoscale_view()  # Rescale the view to fit data
            plt.draw()
            plt.pause(0.01)  # Small pause to create a real-time plotting effect

            iteration += 1

            # Ask if the user wants to continue the simulation or exit
            cont = input("Do you want to continue (y/n)? ").strip().lower()
            if cont != 'y':
                break

    except ValueError:
        print("Please enter valid numeric inputs.")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
