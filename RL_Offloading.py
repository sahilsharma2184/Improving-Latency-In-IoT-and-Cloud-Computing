import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

# Define the Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        """
        Initialize the Double DQN Agent.
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim: Number of possible actions
        - lr: Learning rate for the optimizer
        - gamma: Discount factor for future rewards
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Define main and target Q-networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Synchronize networks

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def choose_action(self, state, epsilon=0.1):
        """
        Choose an action using an epsilon-greedy policy.
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        """
        Perform a Double DQN update step.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([1.0 if done else 0.0])

        # Compute the target Q-value
        with torch.no_grad():
            next_action = torch.argmax(self.q_network(next_state_tensor))
            target_q_value = reward_tensor + self.gamma * (1 - done_tensor) * self.target_network(next_state_tensor)[0, next_action]

        # Compute the current Q-value
        q_value = self.q_network(state_tensor)[0, action_tensor]

        # Compute the loss
        loss = nn.MSELoss()(q_value, target_q_value)

        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """
        Update the target network to match the main Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

# Utility function to validate and adjust state input
def validate_state_input(input_string, state_dim):
    state = list(map(float, input_string.split()))
    if len(state) < state_dim:
        print(f"Warning: Input has fewer than {state_dim} values. Filling missing values with 0.0.")
        state.extend([0.0] * (state_dim - len(state)))
    elif len(state) > state_dim:
        print(f"Warning: Input has more than {state_dim} values. Truncating to {state_dim} values.")
        state = state[:state_dim]
    return state

# Main function for Double DQN with manual input and graph generation
def main():
    try:
        # User input for state and action dimensions
        state_dim = int(input("Enter the dimension of the state space: "))
        action_dim = int(input("Enter the number of possible actions: "))

        # Initialize the Double DQN agent
        agent = DoubleDQNAgent(state_dim, action_dim)

        # User input for the state to track Q-value progression
        track_state = validate_state_input(input(f"Enter the state you want to track (space-separated, {state_dim} values): "), state_dim)

        # Set up the plot for real-time Q-value progression
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        lines = [ax.plot([], [], label=f"Action {a}")[0] for a in range(action_dim)]
        ax.set_xlim(0, 10)  # Set initial x-axis limit
        ax.set_ylim(-1, 10)  # Set y-axis limit
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Q-Value")
        ax.set_title(f"Q-Value Progression for State {track_state}")
        ax.legend()

        iteration = 0
        # Run the Double DQN update in a loop
        while True:
            # Input validation for state
            state = validate_state_input(input(f"Enter the current state (space-separated, {state_dim} values): "), state_dim)

            # Input validation for action
            action = int(input(f"Enter the action taken (0 to {action_dim - 1}): "))
            if action < 0 or action >= action_dim:
                print(f"Error: Action must be between 0 and {action_dim - 1}.")
                continue

            # Reward input
            reward = float(input("Enter the reward: "))

            # Input validation for next state
            next_state = validate_state_input(input(f"Enter the next state (space-separated, {state_dim} values): "), state_dim)

            # Done flag input
            done = input("Is the episode done? (y/n): ").strip().lower() == 'y'

            # Perform the update step
            agent.update(state, action, reward, next_state, done)

            # Update the target network periodically (every 5 iterations)
            if iteration % 5 == 0:
                agent.update_target_network()

            # Display the updated Q-values for the tracked state
            state_tensor = torch.FloatTensor(track_state).unsqueeze(0)
            q_values = agent.q_network(state_tensor).detach().numpy()
            print("Q-Values for the tracked state:", q_values)

            # Update the plot for the tracked state
            for a in range(action_dim):
                # Append new data points for each action in the tracked state
                lines[a].set_xdata(np.append(lines[a].get_xdata(), iteration))
                lines[a].set_ydata(np.append(lines[a].get_ydata(), q_values[0, a]))

            # Extend x-axis limit dynamically to fit the new data
            ax.set_xlim(0, max(10, iteration + 1))

            # Rescale the plot to fit the updated data
            ax.relim()
            ax.autoscale_view()

            # Draw the updated graph and pause briefly to make the update visible
            plt.draw()
            plt.pause(0.5)  # Pause to show the graph for 0.5 seconds after each iteration

            iteration += 1

            # Ask if the user wants to continue
            cont = input("Do you want to continue (y/n)? ").strip().lower()
            if cont != 'y':
                break

    except ValueError:
        print("Please enter valid numeric inputs.")
if __name__ == "__main__":
    main()