import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

class Buffer:
    def __init__(self, state_dim, action_dim, buffer_capacity=100000, batch_size=256):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def sample(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(self.state_buffer[batch_indices])
        action_batch = torch.FloatTensor(self.action_buffer[batch_indices])
        reward_batch = torch.FloatTensor(self.reward_buffer[batch_indices])
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(self.next_state_buffer[batch_indices])
        done_batch = torch.BoolTensor(self.done_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)

        self.lr = actor_lr

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr):
        super(QNetwork, self).__init__()

        self.fc_s = nn.Linear(state_dim, 32)
        self.fc_a = nn.Linear(action_dim, 32)
        self.fc_1 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, action_dim)

        self.lr = critic_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=-1)
        q = F.leaky_relu(self.fc_1(cat))
        q = self.fc_out(q)
        return q


class SAC_Agent:
    def __init__(self):
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-2,2]
        self.lr_pi = 0.001
        self.lr_q = 0.001
        self.gamma = 0.98
        self.batch_size = 200
        self.buffer_limit = 100000
        self.tau = 0.005   # for soft-update of Q using Q-target
        self.init_alpha = 0.01
        self.target_entropy = -1 * self.action_dim
        self.lr_alpha = 0.005
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = Buffer(self.state_dim, self.action_dim, self.buffer_limit, self.batch_size)

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, state):
        with torch.no_grad():
            action, log_prob = self.policy_network.sample(state.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            a_prime, log_prob_prime = self.policy_network.sample(s_prime)
            entropy = - self.log_alpha.exp() * log_prob_prime
            q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            target = r + self.gamma * (1 - done.type(torch.FloatTensor)) * (q_target + entropy)
        return target

    def train_agent(self):
        mini_batch = self.buffer.sample()
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch

        td_target = self.calc_target(mini_batch)

        # Q1 network update
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        self.Q1.optimizer.step()

        # Q2 network update
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        self.Q2.optimizer.step()

        # Policy network update
        a, log_prob = self.policy_network.sample(s_batch)
        entropy = -self.log_alpha.exp() * log_prob

        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        q = torch.min(q1, q2)

        pi_loss = -(q + entropy)  # for gradient ascent
        self.policy_network.optimizer.zero_grad()
        pi_loss.mean().backward()
        self.policy_network.optimizer.step()

        # Alpha parameter update
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Update the target networks
        for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

        # Take the mean of the losses
        q1_loss_val = q1_loss.mean().item()
        q2_loss_val = q2_loss.mean().item()
        pi_loss_val = pi_loss.mean().item()
        alpha_loss_val = alpha_loss.mean().item()

        # Return the average loss values
        return q1_loss_val, q2_loss_val, pi_loss_val, alpha_loss_val


if __name__ == '__main__':

    env = gym.make('Pendulum-v1')
    agent = SAC_Agent()
    total_episodes = 200

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # For storing losses
    q1_losses, q2_losses, pi_losses, alpha_losses = [], [], [], []


    for ep in range(total_episodes):
        state,_ = env.reset()
        episodic_reward = 0
        done = False

        while not done:
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            action = action.detach().cpu().numpy()

            state_prime, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
            else:
                done = False

            agent.buffer.record((state, action, reward, state_prime, done))

            episodic_reward += reward

            state = state_prime

            if agent.buffer.buffer_counter > 1000:
                q1_loss_val, q2_loss_val, pi_loss_val, alpha_loss_val = agent.train_agent()

                # Append the average losses right after training
                q1_losses.append(q1_loss_val)
                q2_losses.append(q2_loss_val)
                pi_losses.append(pi_loss_val)
                alpha_losses.append(alpha_loss_val)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting Avg. Reward
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.title('Avg. Episodic Reward for SAC for Pendulum-v1 Environment')
    plt.show()

    # Plotting Losses
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(q1_losses)
    plt.title("Q1 Losses")
    plt.subplot(2, 2, 2)
    plt.plot(q2_losses)
    plt.title("Q2 Losses")
    plt.subplot(2, 2, 3)
    plt.plot(pi_losses)
    plt.title("Policy Losses")
    plt.subplot(2, 2, 4)
    plt.plot(alpha_losses)
    plt.title("Alpha Losses")
    plt.tight_layout()
    plt.show()

    average_score = sum(ep_reward_list) / total_episodes
    print(f"Average Score over {total_episodes} episodes: {average_score}")

    # Testing Phase
    test_rewards = []
    for ep in range(200):
        state, _ = env.reset()
        episodic_reward = 0
        done = False
        while not done:
            action, _ = agent.choose_action(torch.FloatTensor(state))
            state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            episodic_reward += reward
        test_rewards.append(episodic_reward)

        if ep >= 39:
            # Calculate the average of the last 40 episodes
            avg_reward_last_40 = np.mean(test_rewards[-40:])
            print("Test Episode * {} * Avg Reward of Last 40 Episodes is ==> {}".format(ep, avg_reward_last_40))




    # Plotting Test Rewards for Last 40 Episodes
    plt.plot([np.mean(test_rewards[max(0, i - 39):(i + 1)]) for i in range(len(test_rewards))])
    plt.xlabel("Test Episode")
    plt.ylabel("Avg Reward of Last 40 Episodes")
    plt.title('Average Test Rewards of Last 40 Episodes for SAC on Pendulum-v1 Environment')
    plt.show()

    # Calculate and print the average test score over testing episodes
    average_test_score = sum(test_rewards) / 200
    print(f"Average Test Score over 500 episodes: {average_test_score}")

    env.close()
