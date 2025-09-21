# 📦 Install first if not already:
# pip install gymnasium stable-baselines3 matplotlib

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


# 🎮 Custom Environment
class TreasureEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5):
        super(TreasureEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 4 moves
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )
        self.agent_pos = None
        self.treasure_pos = np.array([self.grid_size - 1, self.grid_size - 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        return self.agent_pos, {}

    def step(self, action):
        if action == 0:   # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2: # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

        reward = -1
        terminated = False
        truncated = False

        if np.array_equal(self.agent_pos, self.treasure_pos):
            reward = 10
            terminated = True

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".")
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"
        grid[self.treasure_pos[0], self.treasure_pos[1]] = "T"
        print("\n".join(" ".join(row) for row in grid))
        print()


# ============================
# 🏗️ Train the RL Agent
# ============================

env = TreasureEnv(grid_size=5)
model = PPO("MlpPolicy", env, verbose=0)

# Track training progress
timesteps = 20000
eval_freq = 1000
rewards = []

print("🚀 Training started...")
for i in range(0, timesteps, eval_freq):
    model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
    
    # Evaluate every few steps
    total_reward = 0
    episodes = 10
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    avg_reward = total_reward / episodes
    rewards.append(avg_reward)
    print(f"Step {i+eval_freq}/{timesteps} | Avg Reward: {avg_reward:.2f}")

print("✅ Training finished!")


# ============================
# 📊 Plot Training Curve
# ============================
plt.plot(range(eval_freq, timesteps + 1, eval_freq), rewards, marker="o")
plt.title("Training Progress (Average Reward over Time)")
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()


# ============================
# 🎯 Final Test Run
# ============================
obs, info = env.reset()
print("\n🎮 Final test run with trained agent:")
for step in range(20):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print("🏆 Reached the treasure!")
        break
