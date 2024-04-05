import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def rule_based_policy(state):
    angular_position = state[0]
    angular_velocity = state[2]

    max_position = 1.0
    max_velocity = 8.0

    position_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    velocity_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    position_intervals = np.linspace(-max_position, max_position, len(position_values) + 1)
    velocity_intervals = np.linspace(-max_velocity, max_velocity, len(velocity_values) + 1)

    position_action = position_values[-1]  # Default action (in case angular_position is at or beyond max_position)
    velocity_action = velocity_values[-1]  # Default action (in case angular_velocity is at or beyond max_velocity)

    for i in range(len(position_values)):
        if position_intervals[i] <= angular_position < position_intervals[i + 1]:
            position_action = position_values[i]
            break  # Breaks out of the for loop once the appropriate action is found

    for i in range(len(velocity_values)):
        if velocity_intervals[i] <= angular_velocity < velocity_intervals[i + 1]:
            velocity_action = velocity_values[i]
            break  # Breaks out of the for loop once the appropriate action is found

    return np.array([position_action, velocity_action])

# Environment setup
env = gym.make("Pendulum-v1")

# Testing Phase
test_rewards = []
for ep in range(300):  # Run for 300 episodes
    state, _ = env.reset()
    episodic_reward = 0
    done = False

    while not done:
        # Ensure state is a NumPy array
        if isinstance(state, tuple):
            state = state[0]
        print(f"Episode {ep + 1}, State: {state}")
        action = rule_based_policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episodic_reward += reward
        state = next_state  # Update state

    test_rewards.append(episodic_reward)
    print(f"Episode {ep + 1} Reward: {episodic_reward}")

# Calculate and print the average test score over testing episodes
average_test_score = sum(test_rewards) / len(test_rewards)
min_test_score = min(test_rewards)
max_test_score = max(test_rewards)

print()
print(f"Average Test Score over {len(test_rewards)} episodes: {average_test_score}")
print(f"Min Test Score: {min_test_score}")
print(f"Max Test Score: {max_test_score}")

window_size = 10
moving_average = np.convolve(test_rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(test_rewards, label='Episodic Reward')
plt.plot(np.arange(len(moving_average)) + window_size//2, moving_average, label=f'Moving Average (Window Size={window_size})', color='orange')
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Test Rewards with Moving Average")
plt.legend()
plt.show()
