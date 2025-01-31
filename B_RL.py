import numpy as np
import matplotlib.pyplot as plt

class BayesianBandit:
    def __init__(self, true_mean):
        """
        Initializes a single bandit (slot machine).
        Parameters:
        - true_mean: The actual mean reward of this bandit (unknown to the agent).
        """
        self.true_mean = true_mean  # Actual mean reward (unknown to agent)
        self.a = 1  # Prior for Beta distribution (successes)
        self.b = 1  # Prior for Beta distribution (failures)

    def pull(self):
        """
        Simulate pulling the bandit's lever and returning a reward.
        """
        return np.random.normal(self.true_mean, 1)  # Assume Gaussian rewards

    def sample(self):
        """
        Sample from the agent's belief about the reward distribution.
        """
        return np.random.beta(self.a, self.b)

    def update(self, reward):
        """
        Update the belief distribution (posterior update) using the observed reward.
        """
        if reward > self.true_mean:
            self.a += 1  # Reward was good, increase "success" count
        else:
            self.b += 1  # Reward was bad, increase "failure" count


class BayesianReinforcementLearning:
    def __init__(self, bandit_means):
        """
        Initializes the Bayesian RL agent.
        Parameters:
        - bandit_means: A list of true mean rewards for each bandit (unknown to the agent).
        """
        self.bandits = [BayesianBandit(mean) for mean in bandit_means]
        self.total_rewards = 0  # Track total rewards
        self.num_pulls = np.zeros(len(bandit_means))  # Track how many times each bandit was pulled

    def select_bandit(self):
        """
        Select a bandit using Thompson Sampling (sample from each belief and pick the best).
        """
        samples = [bandit.sample() for bandit in self.bandits]
        return np.argmax(samples)  # Pick the bandit with the highest sampled value

    def train(self, num_rounds):
        """
        Run the RL loop for a given number of rounds.
        """
        rewards = []
        for _ in range(num_rounds):
            # Select a bandit based on Thompson Sampling
            bandit_index = self.select_bandit()
            
            # Pull the selected bandit
            reward = self.bandits[bandit_index].pull()
            
            # Update belief
            self.bandits[bandit_index].update(reward)
            
            # Track performance
            self.total_rewards += reward
            self.num_pulls[bandit_index] += 1
            rewards.append(self.total_rewards)

        return rewards

# === Running the Bayesian RL Agent ===
true_means = [1.0, 1.5, 2.0]  # Actual reward means for each bandit (unknown to the agent)
agent = BayesianReinforcementLearning(true_means)

# Train the agent for 1000 rounds
rewards = agent.train(1000)

# Plot cumulative rewards
plt.plot(rewards)
plt.xlabel("Rounds")
plt.ylabel("Total Rewards")
plt.title("Bayesian Reinforcement Learning (Thompson Sampling)")
plt.show()

# Print the number of times each bandit was pulled
for i, pulls in enumerate(agent.num_pulls):
    print(f"Bandit {i}: Pulled {int(pulls)} times")
