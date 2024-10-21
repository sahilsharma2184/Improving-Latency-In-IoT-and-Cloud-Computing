import numpy as np

class RLAgent:
    def __init__(self, action_space):
        self.q_table = np.zeros(action_space)  # Initialize Q-table

    def choose_action(self, state):
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, learning_rate=0.1, discount_factor=0.9):
        max_future_q = np.max(self.q_table[state])
        current_q = self.q_table[state, action]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        self.q_table[state, action] = new_q

# Example usage
agent = RLAgent((10, 3))  # 10 states, 3 actions
state = 2
action = agent.choose_action(state)
print(f"Chosen action: {action}")
