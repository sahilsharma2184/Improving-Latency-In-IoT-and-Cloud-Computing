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
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Define main and target Q-networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
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
        self.target_network.load_state_dict(self.q_network.state_dict())

# Simulation parameters
NUM_EPISODES = 500
NUM_DEVICES = 100
BANDWIDTH_LIMIT = 10_000  # kbps

# Initialize RL Agent
state_dim = 5  # [Available Bandwidth, Congestion Level, Latency, Packet Size, Connected Devices]
action_dim = 5  # Different offloading & bandwidth allocation options
agent = DoubleDQNAgent(state_dim, action_dim)

# Storage for Latency Tracking
latency_over_time = []

# Training Loop
for episode in range(NUM_EPISODES):
    # Initial State (Random network conditions)
    bandwidth = BANDWIDTH_LIMIT
    congestion = random.uniform(0.1, 1.0)  
    latency = random.uniform(50, 200)  # Initial latency in ms
    packet_size = random.randint(500, 1500)  
    num_connected_devices = random.randint(50, NUM_DEVICES)

    state = [bandwidth, congestion, latency, packet_size, num_connected_devices]

    # Fix Epsilon Decay: Ensure enough exploration
    epsilon = max(0.01, 1.0 - episode / (NUM_EPISODES * 0.75))

    # Fix Learning Rate Decay: Prevent stagnation
    lr = max(0.0001, 0.005 * (1 - episode / (NUM_EPISODES * 1.5)))
    agent.optimizer = optim.Adam(agent.q_network.parameters(), lr=lr)

    for step in range(50):  
        action = agent.choose_action(state, epsilon)

        if action == 0:
            bandwidth += 500
        elif action == 1:
            congestion -= 0.05
        elif action == 2:
            packet_size *= 0.9
        elif action == 3:
            num_connected_devices *= 0.9
        elif action == 4:
            latency *= 0.8  

        new_latency = (packet_size / bandwidth) * 1000 + congestion * latency
        latency_change = abs(new_latency - latency)

        # Fix Reward Function: More reward for stable low latency, strong penalty for spikes
        reward = -new_latency  
        if 80 <= new_latency <= 90:
            reward += 50  # Stronger reward for ideal range
        elif new_latency < 80:
            reward += 20  # Reward for further reduction
        elif new_latency > 100:
            reward -= 70  # Stronger penalty for high latency
        reward -= 0.3 * latency_change  # Reduce sudden fluctuations

        latency_over_time.append(new_latency)
        next_state = [bandwidth, congestion, new_latency, packet_size, num_connected_devices]

        # Fix Done Condition: Don't exit too early
        done = new_latency < 70  

        agent.update(state, action, reward, next_state, done)
        state = next_state

        if done:
            break

    if episode % 10 == 0:
        agent.update_target_network()

    # **Fix Terminal Output**
    print(f"Episode {episode}: Latency = {new_latency:.2f} ms, Epsilon = {epsilon:.3f}, Learning Rate = {lr:.5f}")

# Plot Smoother Graph
smoothed_latency = np.convolve(latency_over_time, np.ones(500)/500, mode='valid')
plt.plot(smoothed_latency, label="Smoothed Latency Over Time", color='r')
plt.xlabel("Steps")
plt.ylabel("Network Latency (ms)")
plt.title("Final Optimized RL-Based IoT Network Latency Minimization")
plt.legend()
plt.grid(True)
plt.show()
