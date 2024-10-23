import numpy as np

class RLAgent:
    def __init__(self, num_states, num_actions):
        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))  

    def choose_action(self, state):
        # Choose the action with the highest Q-value for the given state
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, learning_rate=0.1, discount_factor=0.9):
        # Q-value update formula
        max_future_q = np.max(self.q_table[state])  # Best future Q-value
        current_q = self.q_table[state, action]  # Current Q-value
        # New Q-value calculation
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        self.q_table[state, action] = new_q  # Update Q-table

def main():
    try:
        # User input: number of states and actions
        num_states = int(input("Enter the number of states: "))
        num_actions = int(input("Enter the number of actions: "))

        # Initialize the RL agent
        agent = RLAgent(num_states, num_actions)

        # Run multiple iterations to update the Q-table dynamically
        while True:
            # User inputs for state, action, and reward
            state = int(input(f"Enter the state (0 to {num_states - 1}): "))
            action = int(input(f"Enter the action (0 to {num_actions - 1}): "))
            reward = float(input("Enter the reward: "))

            # Agent chooses an action based on the current state
            chosen_action = agent.choose_action(state)
            print(f"Chosen action for state {state}: {chosen_action}")

            # Update the Q-value with the provided state, action, and reward
            agent.update_q_value(state, action, reward)

            # Display the updated Q-table
            print("Updated Q-Table:")
            print(agent.q_table)

            # Ask if the user wants to continue or exit
            cont = input("Do you want to continue (y/n)? ").strip().lower()
            if cont != 'y':
                break

    except ValueError:
        print("Please enter valid numeric inputs.")

# Run the main function
if __name__ == "__main__":
    main()
