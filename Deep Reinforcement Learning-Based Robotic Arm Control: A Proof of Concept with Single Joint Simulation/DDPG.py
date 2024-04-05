import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, state_size, action_size, upper_bound):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_size)
        self.upper_bound = upper_bound

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.upper_bound
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_layer1 = nn.Linear(state_size, 16)
        self.state_layer2 = nn.Linear(16, 32)

        self.action_layer1 = nn.Linear(action_size, 32)

        self.concat_layer1 = nn.Linear(64, 256)
        self.concat_layer2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, state, action):
        state_out = torch.relu(self.state_layer1(state))
        state_out = torch.relu(self.state_layer2(state_out))

        action_out = torch.relu(self.action_layer1(action))

        concat = torch.cat([state_out, action_out], 1)

        out = torch.relu(self.concat_layer1(concat))
        out = torch.relu(self.concat_layer2(out))
        return self.output_layer(out)
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def learn(self, actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, gamma,
              tau):
        # Initialize losses as None
        actor_loss = None
        critic_loss = None

        if self.buffer_counter >= self.batch_size:
            indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), size=self.batch_size)

            state_batch = torch.tensor(self.state_buffer[indices], dtype=torch.float)
            action_batch = torch.tensor(self.action_buffer[indices], dtype=torch.float)
            reward_batch = torch.tensor(self.reward_buffer[indices], dtype=torch.float)
            next_state_batch = torch.tensor(self.next_state_buffer[indices], dtype=torch.float)

            # Critic loss
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic(next_state_batch, target_actions).detach()
            critic_value = critic_model(state_batch, action_batch)
            critic_loss = nn.MSELoss()(critic_value, y)

            # Backpropagation for critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor loss
            new_actions = actor_model(state_batch)
            actor_loss = -critic_model(state_batch, new_actions).mean()

            # Backpropagation for actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target networks
            self.update_target_network(target_actor, actor_model, tau)
            self.update_target_network(target_critic, critic_model, tau)

        return actor_loss, critic_loss

    @staticmethod
    def update_target_network(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def policy(state, actor_model, noise_object, lower_bound, upper_bound):
    state = torch.from_numpy(state).float()
    action = actor_model(state).detach().numpy()
    if noise_object is not None:
        action += noise_object()
    action = np.clip(action, lower_bound, upper_bound)
    return action

# Environment setup
problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# Noise process
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Actor-Critic Models
actor_model = Actor(num_states, num_actions, upper_bound)
critic_model = Critic(num_states, num_actions)
target_actor = Actor(num_states, num_actions, upper_bound)
target_critic = Critic(num_states, num_actions)

# Making the weights equal initially
target_actor.load_state_dict(actor_model.state_dict())
target_critic.load_state_dict(critic_model.state_dict())

# Optimizers
critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.002)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)

# Hyperparameters
total_episodes = 500
gamma = 0.99
tau = 0.005
buffer = Buffer(50000, 64)

# Training loop
ep_reward_list = []
avg_reward_list = []
actor_loss_list = []
critic_loss_list = []

for ep in range(total_episodes):
    state, _ = env.reset()
    episodic_reward = 0

    while True:
        action = policy(state, actor_model, ou_noise, lower_bound, upper_bound)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.record((state, action, reward, next_state))
        episodic_reward += reward

        actor_loss, critic_loss = buffer.learn(actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, gamma, tau)

        if actor_loss is not None and critic_loss is not None:
            # Append losses to their respective lists
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())


        state = next_state
        if done:
            break

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)



# Plotting graph
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

'''
torch.save(actor_model.state_dict(), 'ddpg_actor_network.pt')
torch.save(critic_model.state_dict(), 'ddpg_critic_network.pt')
'''


# Plotting Losses (assuming you have tracked actor_loss_list and critic_loss_list during training)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(actor_loss_list)
plt.title("Actor Losses")
plt.subplot(1, 2, 2)
plt.plot(critic_loss_list)
plt.title("Critic Losses")
plt.tight_layout()
plt.show()

average_score = sum(ep_reward_list) / total_episodes
print(f"Average Score over {total_episodes} episodes: {average_score}")


# Testing Phase
test_rewards = []
for ep in range(500):
    state, _ = env.reset()
    episodic_reward = 0
    done = False
    while not done:
        # Directly use the policy without noise for testing
        action = policy(state, actor_model, None, lower_bound, upper_bound)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episodic_reward += reward
    test_rewards.append(episodic_reward)

    if ep >= 39:
        avg_reward_last_40 = np.mean(test_rewards[-40:])
        print("Test Episode * {} * Avg Reward of Last 40 Episodes is ==> {}".format(ep, avg_reward_last_40))



# Plotting Test Rewards for Last 40 Episodes
plt.plot([np.mean(test_rewards[max(0, i - 39):(i + 1)]) for i in range(len(test_rewards))])
plt.xlabel("Test Episode")
plt.ylabel("Avg Reward of Last 40 Episodes")
plt.title('Average Test Rewards of Last 40 Episodes for DDPG on Pendulum-v1 Environment')
plt.show()

# Calculate and print the average test score over testing episodes
average_test_score = sum(test_rewards) / 500  # or len(test_rewards)
print(f"Average Test Score over 500 episodes: {average_test_score}")

env.close()


