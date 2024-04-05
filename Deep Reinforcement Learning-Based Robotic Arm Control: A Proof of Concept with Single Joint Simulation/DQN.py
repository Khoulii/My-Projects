import matplotlib.pyplot as plt
import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Neural Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_new = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x) * 2  # Apply Tanh and scale the output to [-2, 2]
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=256, gamma=0.99, epsilon=1, buffer_size=10000):
        self.model = QNetwork(state_size, hidden_size, action_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Random action in the range of the action space
            return [4 * random.random() - 2]  # Action range [-2, 2]
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                action = self.model(state).item()  # Directly use the action from the model
                return [action]

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn_from_batch(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Directly predict actions from the current states
        predicted_actions = self.model(states)

        # Define the loss function
        # Here, we want to maximize rewards, so we minimize the negative reward
        # We can use a simple mean squared error between predicted and actual actions weighted by the negative reward
        loss = ((predicted_actions - actions) ** 2 * (-rewards)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_epsilon(self, decay_rate):
        self.epsilon *= decay_rate

# Environment setup
env = gym.make("Pendulum-v1")

num_episodes = 500
total_scores = []
cumulative_rewards = []
losses = []
epsilons = []
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = DQNAgent(state_size, action_size)
batch_size = 512

# Training loop
ep_reward_list = []
avg_reward_list = []

for episode in range(num_episodes):
    observation, info = env.reset()
    total_reward = 0
    cumulative_reward = 0
    state = observation

    while True:
        action = agent.choose_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = observation

        agent.push_to_buffer(state, action, reward, next_state, terminated or truncated)
        if len(agent.replay_buffer) > batch_size:
            loss = agent.learn_from_batch(batch_size)
            if loss is not None:
                losses.append(loss.item())

        total_reward += reward
        cumulative_reward += reward
        state = next_state

        if terminated or truncated:
            break

    # After an episode ends, update and record the epsilon value
    agent.update_epsilon(0.999)
    epsilons.append(agent.epsilon)

    # Recording rewards and other metrics for the episode
    ep_reward_list.append(total_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()

# Plotting
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

# Plotting losses
plt.subplot(2, 1, 1)  # First subplot
plt.plot(losses)
plt.title("Loss per Training Step")

# Plotting epsilon values
plt.subplot(2, 1, 2)  # Second subplot
plt.plot(epsilons)
plt.title("Epsilon Values per Episode")

plt.tight_layout()
plt.show()

# Calculating average score
average_score = sum(ep_reward_list) / num_episodes
print(f"Average Score over {num_episodes} episodes: {average_score}")

# Function to test the DQN agent remains the same
def test_dqn_agent(env, agent, num_test_episodes):
    test_scores = []

    for episode in range(num_test_episodes):
        observation, _ = env.reset()
        total_reward = 0
        state = observation

        while True:
            action = agent.choose_action(state)  # Choose action based on the neural network
            observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        test_scores.append(total_reward)


    return test_scores

# Set the number of test episodes
num_test_episodes = 500

# Temporarily set epsilon to 0 to disable exploration during testing
original_epsilon = agent.epsilon
agent.epsilon = 0

# Test the agent
test_scores = test_dqn_agent(env, agent, num_test_episodes)

# Restore the original epsilon value
agent.epsilon = original_epsilon

plt.plot([np.mean(test_scores[max(0, i - 39):(i + 1)]) for i in range(len(test_scores))])
plt.xlabel("Test Episode")
plt.ylabel("Avg Reward of Last 40 Episodes")
plt.title('Average Test Rewards of Last 40 Episodes for DQN on Pendulum-v1 Environment')
plt.show()

# Calculate and print the average test score over testing episodes
average_test_score = sum(test_scores) / 500  # or len(test_rewards)
print(f"Average Test Score over 500 episodes: {average_test_score}")

env.close()
