import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# -----------------------------
# 1. Maze Environment with Pygame
# -----------------------------
class MazeEnv:
    def __init__(self, size=8, cell_size=60):
        self.size = size
        self.cell_size = cell_size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.agent_pos = self.start

        # Complex walls for challenge
        self.walls = {
            (1,1),(1,2),(1,3),(2,3),(3,3),(4,1),(4,2),(4,3),(5,5),(5,6),(6,5)
        }

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((size * cell_size, size * cell_size))
        pygame.display.set_caption("RL Maze Learning")

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.size - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.size - 1, y + 1)

        if (x, y) in self.walls:  # can't move into walls
            x, y = self.agent_pos

        self.agent_pos = (x, y)

        # Reward system
        if self.agent_pos == self.goal:
            return self.agent_pos, 10, True
        else:
            return self.agent_pos, -1, False

    def render(self, episode=None, step=None, total_reward=None):
        self.screen.fill((255, 255, 255))  # white background

        # Draw grid
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200,200,200), rect, 1)

        # Draw walls
        for (i,j) in self.walls:
            rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0,0,0), rect)

        # Draw goal
        gx, gy = self.goal
        rect = pygame.Rect(gy*self.cell_size, gx*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0,255,0), rect)

        # Draw agent
        ax, ay = self.agent_pos
        rect = pygame.Rect(ay*self.cell_size, ax*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0,0,255), rect)

        # Info text
        if episode is not None:
            font = pygame.font.SysFont("Arial", 20)
            text = font.render(f"Ep:{episode} Step:{step} Reward:{total_reward}", True, (0,0,0))
            self.screen.blit(text, (10,10))

        pygame.display.flip()
        pygame.time.delay(50)

# -----------------------------
# 2. Q-learning Agent
# -----------------------------
class QLearningAgent:
    def __init__(self, env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 4))  # states: (x,y), actions:4
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rewards = []

    def train(self):
        for ep in range(1, self.episodes+1):
            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                # Handle quitting
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                x, y = state

                # ε-greedy
                if random.uniform(0,1) < self.epsilon:
                    action = np.random.choice(4)
                else:
                    action = np.argmax(self.q_table[x, y])

                next_state, reward, done = self.env.step(action)
                nx, ny = next_state

                # Update Q-value
                old_value = self.q_table[x, y, action]
                next_max = np.max(self.q_table[nx, ny])
                new_value = (1-self.alpha)*old_value + self.alpha*(reward + self.gamma*next_max)
                self.q_table[x, y, action] = new_value

                state = next_state
                total_reward += reward
                step += 1

                # Render maze every few episodes for speed
                if ep % 50 == 0:
                    self.env.render(ep, step, total_reward)

            self.rewards.append(total_reward)
        print("Training finished ✅")

    def test(self):
        state = self.env.reset()
        done = False
        path = [state]
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            x, y = state
            action = np.argmax(self.q_table[x, y])
            state, reward, done = self.env.step(action)
            path.append(state)
            total_reward += reward
            self.env.render("TEST", "-", total_reward)
            pygame.time.delay(100)
        print(f"Test finished! Total reward: {total_reward}")
        print(f"Path taken: {path}")

        return path

# -----------------------------
# 3. Run Everything
# -----------------------------
if __name__ == "__main__":
    env = MazeEnv(size=8)
    agent = QLearningAgent(env, episodes=500)
    agent.train()

    # Plot training rewards
    plt.plot(agent.rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.show()

    # Test final policy
    agent.test()

    pygame.quit()
