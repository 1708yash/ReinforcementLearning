#!/usr/bin/env python3
"""
Enhanced Reinforcement Learning Game with StableBaseline3
========================================================

🎮 Advanced RL Game featuring:
- Multiple RL algorithms (PPO, DQN, A2C, SAC, TD3)
- Continuous visual updates during training
- Enhanced control features and real-time analytics
- JSON analytics storage and data export
- Robust error handling and threading
- Professional GUI with advanced features
- Python 3.10+ compatible

Author: AI Assistant (Enhanced Version)
Date: September 2025

Requirements:
pip install stable-baselines3[extra] gymnasium numpy matplotlib torch tensorboard json5

Usage:
python enhanced_rl_game.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
import threading
import time
import queue
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import os
import json
import datetime
from typing import Any, Dict, Tuple, Optional, List
import warnings
import sys
from collections import deque
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

class GridWorldEnv(gym.Env):
    """
    Enhanced 2D GridWorld Environment with improved features
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, size: int = 10, obstacle_density: float = 0.15, 
                 dynamic_obstacles: bool = False, reward_shaping: bool = True):
        super().__init__()

        self.size = size
        self.obstacle_density = obstacle_density
        self.dynamic_obstacles = dynamic_obstacles
        self.reward_shaping = reward_shaping

        # Enhanced observation space with additional features
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
            'goal': spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
            'obstacles': spaces.Box(0, 1, shape=(size, size), dtype=np.int32),
            'visited': spaces.Box(0, 1, shape=(size, size), dtype=np.int32),
            'distance_to_goal': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
        })

        # Enhanced action space
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]),  # up
            1: np.array([0, 1]),   # right
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),  # left
        }

        # Initialize state
        self._agent_location = np.array([0, 0], dtype=np.int32)
        self._goal_location = np.array([size-1, size-1], dtype=np.int32)
        self._obstacles = np.zeros((size, size), dtype=np.int32)
        self._visited = np.zeros((size, size), dtype=np.int32)

        # Performance tracking
        self.episode_step_count = 0
        self.max_episode_steps = size * size  # Dynamic based on grid size
        self.total_episodes = 0

        # Animation state
        self.render_history = []

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get enhanced observation with additional features"""
        distance = np.linalg.norm(self._agent_location - self._goal_location)
        return {
            'agent': self._agent_location.copy(),
            'goal': self._goal_location.copy(),
            'obstacles': self._obstacles.copy(),
            'visited': self._visited.copy(),
            'distance_to_goal': np.array([distance], dtype=np.float32)
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information"""
        distance = np.linalg.norm(self._agent_location - self._goal_location)
        manhattan_dist = np.sum(np.abs(self._agent_location - self._goal_location))
        visited_count = np.sum(self._visited)

        return {
            'distance': float(distance),
            'manhattan_distance': int(manhattan_dist),
            'steps': self.episode_step_count,
            'visited_cells': int(visited_count),
            'exploration_ratio': float(visited_count / (self.size * self.size)),
            'episode': self.total_episodes,
            'optimal_steps': int(manhattan_dist)  # Minimum steps needed
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Enhanced reset with better initialization"""
        super().reset(seed=seed)

        self.episode_step_count = 0
        self.total_episodes += 1

        # Smart agent and goal placement
        self._agent_location = self.np_random.integers(0, max(1, self.size//3), size=2, dtype=np.int32)
        self._goal_location = self.np_random.integers(
            max(self.size*2//3, 1), self.size, size=2, dtype=np.int32
        )

        # Create strategic obstacles
        self._obstacles = np.zeros((self.size, self.size), dtype=np.int32)
        self._visited = np.zeros((self.size, self.size), dtype=np.int32)

        num_obstacles = max(1, int(self.size * self.size * self.obstacle_density))

        # Create obstacle clusters for more interesting navigation
        for _ in range(max(1, num_obstacles // 3)):
            center = self.np_random.integers(1, self.size-1, size=2)
            cluster_size = self.np_random.integers(1, 4)

            for _ in range(cluster_size):
                obs_pos = center + self.np_random.integers(-1, 2, size=2)
                obs_pos = np.clip(obs_pos, 0, self.size-1)

                if not (np.array_equal(obs_pos, self._agent_location) or 
                       np.array_equal(obs_pos, self._goal_location)):
                    self._obstacles[obs_pos[0], obs_pos[1]] = 1

        # Mark starting position as visited
        self._visited[self._agent_location[0], self._agent_location[1]] = 1

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Enhanced step function with improved reward shaping"""
        self.episode_step_count += 1

        old_distance = np.linalg.norm(self._agent_location - self._goal_location)
        old_position = self._agent_location.copy()

        # Execute action
        direction = self._action_to_direction[action]
        new_position = self._agent_location + direction

        # Check bounds
        new_position = np.clip(new_position, 0, self.size - 1)

        # Check for obstacles
        hit_obstacle = False
        if self._obstacles[new_position[0], new_position[1]]:
            hit_obstacle = True
            new_position = old_position  # Stay in place

        self._agent_location = new_position

        # Update visited cells
        self._visited[self._agent_location[0], self._agent_location[1]] = 1

        # Enhanced reward calculation
        reward = 0.0
        terminated = False

        if np.array_equal(self._agent_location, self._goal_location):
            # Goal reached - large reward with efficiency bonus
            base_reward = 100.0
            efficiency_bonus = max(0, 50 - self.episode_step_count)
            reward = base_reward + efficiency_bonus
            terminated = True
        else:
            if self.reward_shaping:
                # Distance-based reward shaping
                new_distance = np.linalg.norm(self._agent_location - self._goal_location)
                distance_reward = (old_distance - new_distance) * 5.0

                # Exploration bonus
                exploration_bonus = 0.5 if self._visited[self._agent_location[0], self._agent_location[1]] == 1 else 1.0

                # Penalties
                step_penalty = -0.1
                obstacle_penalty = -2.0 if hit_obstacle else 0.0

                reward = distance_reward + exploration_bonus + step_penalty + obstacle_penalty
            else:
                reward = -0.1  # Simple step penalty
                if hit_obstacle:
                    reward -= 1.0

        # Check for truncation
        truncated = self.episode_step_count >= self.max_episode_steps

        # Store render data for animation
        self.render_history.append({
            'agent': self._agent_location.copy(),
            'goal': self._goal_location.copy(),
            'obstacles': self._obstacles.copy(),
            'visited': self._visited.copy(),
            'reward': reward,
            'step': self.episode_step_count
        })

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render_to_array(self, show_visited: bool = True) -> np.ndarray:
        """Enhanced rendering with visited cells visualization"""
        rgb_array = np.ones((self.size, self.size, 3), dtype=np.float32)

        # Show visited cells as light gray
        if show_visited:
            visited_positions = np.where(self._visited == 1)
            rgb_array[visited_positions[0], visited_positions[1]] = [0.9, 0.9, 0.9]

        # Draw obstacles (black)
        obstacle_positions = np.where(self._obstacles == 1)
        rgb_array[obstacle_positions[0], obstacle_positions[1]] = [0.1, 0.1, 0.1]

        # Draw goal (red with pulsing effect based on distance)
        goal_intensity = 0.7 + 0.3 * np.sin(time.time() * 5)
        rgb_array[self._goal_location[0], self._goal_location[1]] = [1, 0, 0]

        # Draw agent (blue)
        rgb_array[self._agent_location[0], self._agent_location[1]] = [0, 0, 1]

        return rgb_array

class FlattenDictWrapper(gym.ObservationWrapper):
    """Enhanced wrapper for flattening Dict observations"""

    def __init__(self, env):
        super().__init__(env)

        # Calculate flattened observation space size
        total_size = 0
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                total_size += np.prod(space.shape)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )

        # Store key order for consistent flattening
        self.key_order = ['agent', 'goal', 'obstacles', 'visited', 'distance_to_goal']

    def observation(self, obs):
        """Flatten observation maintaining consistent order"""
        flat_obs = []
        for key in self.key_order:
            if key in obs:
                if obs[key].ndim == 0:
                    flat_obs.append(obs[key])
                else:
                    flat_obs.extend(obs[key].flatten())

        return np.array(flat_obs, dtype=np.float32)

class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback with comprehensive monitoring"""

    def __init__(self, gui_app, log_interval: int = 100):
        super().__init__()
        self.gui_app = gui_app
        self.log_interval = log_interval

        # Enhanced tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.training_losses = []
        self.exploration_rates = []

        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_success = False
        self.episode_count = 0

        # Performance metrics
        self.best_reward = float('-inf')
        self.success_rate_window = deque(maxlen=50)
        self.reward_window = deque(maxlen=100)

        # Analytics data for JSON export
        self.analytics_data = {
            'training_start_time': datetime.datetime.now().isoformat(),
            'episodes': [],
            'training_config': {},
            'performance_metrics': {}
        }

    def _on_training_start(self) -> None:
        """Called at the beginning of training"""
        self.analytics_data['training_config'] = {
            'algorithm': self.gui_app.algo_var.get(),
            'total_timesteps': int(self.gui_app.steps_var.get()),
            'learning_rate': float(self.gui_app.lr_var.get()),
            'environment_size': int(self.gui_app.size_var.get())
        }

    def _on_step(self) -> bool:
        """Enhanced step monitoring"""
        if not self.gui_app.is_training:
            return False

        # Get current step info
        reward = self.locals.get('rewards', [0])[0]
        info = self.locals.get('infos', [{}])[0]
        done = self.locals.get('dones', [False])[0]

        # Update current episode
        self.current_episode_reward += reward
        self.current_episode_length += 1

        # Check for goal achievement
        if reward > 50:  # Goal reached
            self.current_episode_success = True

        # Episode ended
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_successes.append(self.current_episode_success)

            # Update windows
            self.reward_window.append(self.current_episode_reward)
            self.success_rate_window.append(1.0 if self.current_episode_success else 0.0)

            # Update best reward
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward

            # Calculate metrics
            mean_reward = np.mean(list(self.reward_window))
            success_rate = np.mean(list(self.success_rate_window)) * 100

            # Store episode data for JSON
            episode_data = {
                'episode': self.episode_count,
                'reward': float(self.current_episode_reward),
                'length': self.current_episode_length,
                'success': self.current_episode_success,
                'mean_reward': float(mean_reward),
                'success_rate': float(success_rate),
                'timestamp': datetime.datetime.now().isoformat()
            }

            if info:
                episode_data.update({
                    'exploration_ratio': info.get('exploration_ratio', 0),
                    'optimal_steps': info.get('optimal_steps', 0)
                })

            self.analytics_data['episodes'].append(episode_data)

            # Update GUI
            self.gui_app.update_training_stats(
                episode=self.episode_count,
                mean_reward=mean_reward,
                last_reward=self.current_episode_reward,
                episode_length=self.current_episode_length,
                success_rate=success_rate,
                best_reward=self.best_reward
            )

            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_success = False

        return True

    def _on_training_end(self) -> None:
        """Called when training ends"""
        self.analytics_data['training_end_time'] = datetime.datetime.now().isoformat()
        self.analytics_data['performance_metrics'] = {
            'total_episodes': self.episode_count,
            'best_reward': float(self.best_reward),
            'final_success_rate': float(np.mean(list(self.success_rate_window)) * 100) if self.success_rate_window else 0,
            'final_mean_reward': float(np.mean(list(self.reward_window))) if self.reward_window else 0
        }

        # Save analytics to JSON file
        self.gui_app.save_analytics_to_json(self.analytics_data)

class EnhancedRLGameApp:
    """Enhanced RL Game Application with advanced features"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Enhanced RL Game - Professional Edition")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)
        self.root.configure(bg='#2b2b2b')

        # Enhanced variables
        self.env = None
        self.model = None
        self.is_training = False
        self.training_thread = None
        self.visualization_thread = None
        self.visualization_active = True

        # Animation control
        self.game_animation = None
        self.analytics_animation = None
        self.update_queue = queue.Queue()

        # Training statistics
        self.training_stats = {
            'episodes': [],
            'mean_rewards': [],
            'last_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'best_rewards': []
        }

        # Style configuration
        self.setup_styles()
        self.setup_gui()
        self.setup_plots()
        self.start_continuous_updates()

        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Setup enhanced visual styles"""
        style = ttk.Style()

        # Configure styles
        style.theme_use('clam')

        # Custom button styles
        style.configure('Success.TButton', foreground='white', background='#28a745')
        style.configure('Danger.TButton', foreground='white', background='#dc3545')
        style.configure('Warning.TButton', foreground='white', background='#ffc107')
        style.configure('Primary.TButton', foreground='white', background='#007bff')

    def setup_gui(self):
        """Setup enhanced GUI with professional layout"""

        # Main container with dark theme
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Create paned window for resizable layout
        main_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left control panel (enhanced)
        control_frame = ttk.Frame(main_paned)
        main_paned.add(control_frame, weight=1)

        control_panel = ttk.LabelFrame(control_frame, text="🎛️ Advanced Control Panel", padding="15")
        control_panel.pack(fill=tk.BOTH, expand=True)

        # Environment Configuration
        self.setup_environment_controls(control_panel)

        # Algorithm Selection  
        self.setup_algorithm_controls(control_panel)

        # Training Controls
        self.setup_training_controls(control_panel)

        # Advanced Features
        self.setup_advanced_controls(control_panel)

        # Statistics Display
        self.setup_statistics_display(control_panel)

        # Right visualization panel
        viz_frame = ttk.Frame(main_paned)
        main_paned.add(viz_frame, weight=3)

        # Enhanced tabbed interface
        self.setup_visualization_tabs(viz_frame)

    def setup_environment_controls(self, parent):
        """Setup environment configuration controls"""
        env_frame = ttk.LabelFrame(parent, text="🌍 Environment Configuration", padding="10")
        env_frame.pack(fill=tk.X, pady=(0, 10))

        # Grid configuration
        config_frame = ttk.Frame(env_frame)
        config_frame.pack(fill=tk.X, pady=5)

        ttk.Label(config_frame, text="Grid Size:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky='w')
        self.size_var = tk.StringVar(value="12")
        size_spinbox = ttk.Spinbox(config_frame, from_=5, to=25, textvariable=self.size_var, width=8)
        size_spinbox.grid(row=0, column=1, padx=(5, 10))

        ttk.Label(config_frame, text="Obstacle Density:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky='w')
        self.density_var = tk.StringVar(value="0.15")
        density_spinbox = ttk.Spinbox(config_frame, from_=0.05, to=0.4, increment=0.05, 
                                     textvariable=self.density_var, width=8)
        density_spinbox.grid(row=1, column=1, padx=(5, 10))

        # Environment features
        features_frame = ttk.Frame(env_frame)
        features_frame.pack(fill=tk.X, pady=5)

        self.reward_shaping_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_frame, text="Reward Shaping", 
                       variable=self.reward_shaping_var).pack(side=tk.LEFT)

        self.show_visited_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_frame, text="Show Visited Cells", 
                       variable=self.show_visited_var).pack(side=tk.LEFT, padx=(10, 0))

        # Create environment button
        self.create_env_btn = ttk.Button(env_frame, text="🔄 Create Environment", 
                                        command=self.create_environment, 
                                        style="Primary.TButton")
        self.create_env_btn.pack(fill=tk.X, pady=(10, 0))

    def setup_algorithm_controls(self, parent):
        """Setup algorithm selection and configuration"""
        algo_frame = ttk.LabelFrame(parent, text="🧠 Algorithm Configuration", padding="10")
        algo_frame.pack(fill=tk.X, pady=(0, 10))

        # Algorithm selection
        ttk.Label(algo_frame, text="Algorithm:", font=("Arial", 10, "bold")).pack(anchor='w')
        self.algo_var = tk.StringVar(value="PPO")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var,
                                 values=["PPO", "A2C", "DQN", "SAC", "TD3"], 
                                 state="readonly", width=25)
        algo_combo.pack(fill=tk.X, pady=(2, 10))

        # Training parameters
        params_frame = ttk.Frame(algo_frame)
        params_frame.pack(fill=tk.X)

        # Training steps
        ttk.Label(params_frame, text="Training Steps:", font=("Arial", 9)).grid(row=0, column=0, sticky='w')
        self.steps_var = tk.StringVar(value="50000")
        steps_entry = ttk.Entry(params_frame, textvariable=self.steps_var, width=12)
        steps_entry.grid(row=0, column=1, padx=(5, 10))

        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:", font=("Arial", 9)).grid(row=1, column=0, sticky='w')
        self.lr_var = tk.StringVar(value="0.0003")
        lr_entry = ttk.Entry(params_frame, textvariable=self.lr_var, width=12)
        lr_entry.grid(row=1, column=1, padx=(5, 10))

        # Batch size
        ttk.Label(params_frame, text="Batch Size:", font=("Arial", 9)).grid(row=0, column=2, sticky='w')
        self.batch_size_var = tk.StringVar(value="64")
        batch_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10)
        batch_entry.grid(row=0, column=3, padx=(5, 0))

        # Gamma (discount factor)
        ttk.Label(params_frame, text="Gamma:", font=("Arial", 9)).grid(row=1, column=2, sticky='w')
        self.gamma_var = tk.StringVar(value="0.99")
        gamma_entry = ttk.Entry(params_frame, textvariable=self.gamma_var, width=10)
        gamma_entry.grid(row=1, column=3, padx=(5, 0))

    def setup_training_controls(self, parent):
        """Setup training control buttons"""
        training_frame = ttk.LabelFrame(parent, text="🚀 Training Controls", padding="10")
        training_frame.pack(fill=tk.X, pady=(0, 10))

        # Main control buttons
        control_buttons = ttk.Frame(training_frame)
        control_buttons.pack(fill=tk.X)

        self.train_btn = ttk.Button(control_buttons, text="🚀 Start Training", 
                                   command=self.start_training, state="disabled",
                                   style="Success.TButton")
        self.train_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.stop_train_btn = ttk.Button(control_buttons, text="⏹️ Stop Training", 
                                        command=self.stop_training, state="disabled",
                                        style="Danger.TButton")
        self.stop_train_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.pause_btn = ttk.Button(control_buttons, text="⏸️ Pause", 
                                   command=self.pause_training, state="disabled",
                                   style="Warning.TButton")
        self.pause_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Game control buttons
        game_buttons = ttk.Frame(training_frame)
        game_buttons.pack(fill=tk.X, pady=(10, 0))

        self.play_random_btn = ttk.Button(game_buttons, text="🎲 Random Agent", 
                                         command=self.play_random_agent, state="disabled")
        self.play_random_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.play_trained_btn = ttk.Button(game_buttons, text="🧠 Trained Agent", 
                                          command=self.play_trained_model, state="disabled",
                                          style="Success.TButton")
        self.play_trained_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var,
                                           mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))

    def setup_advanced_controls(self, parent):
        """Setup advanced control features"""
        advanced_frame = ttk.LabelFrame(parent, text="⚙️ Advanced Features", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))

        # Model management
        model_frame = ttk.Frame(advanced_frame)
        model_frame.pack(fill=tk.X, pady=5)

        self.save_model_btn = ttk.Button(model_frame, text="💾 Save Model", 
                                        command=self.save_model, state="disabled")
        self.save_model_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.load_model_btn = ttk.Button(model_frame, text="📂 Load Model", 
                                        command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.export_data_btn = ttk.Button(model_frame, text="📊 Export Data", 
                                         command=self.export_analytics_data)
        self.export_data_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Visualization controls
        viz_controls = ttk.Frame(advanced_frame)
        viz_controls.pack(fill=tk.X, pady=5)

        self.animation_speed_var = tk.DoubleVar(value=0.2)
        ttk.Label(viz_controls, text="Animation Speed:", font=("Arial", 9)).pack(side=tk.LEFT)
        speed_scale = ttk.Scale(viz_controls, from_=0.05, to=1.0, orient=tk.HORIZONTAL,
                               variable=self.animation_speed_var, length=100)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))

        self.realtime_viz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_controls, text="Real-time Viz", 
                       variable=self.realtime_viz_var).pack(side=tk.RIGHT)

    def setup_statistics_display(self, parent):
        """Setup enhanced statistics display"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Training Statistics", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for different stat views
        stats_notebook = ttk.Notebook(stats_frame)
        stats_notebook.pack(fill=tk.BOTH, expand=True)

        # Current stats tab
        current_tab = ttk.Frame(stats_notebook)
        stats_notebook.add(current_tab, text="Current")

        self.stats_text = tk.Text(current_tab, height=12, font=("Consolas", 9),
                                 bg='#f8f8f8', state='disabled', wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(current_tab, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status display
        status_frame = ttk.Frame(stats_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(status_frame, text="Status: Ready to begin 🚀", 
                                     foreground="#28a745", font=("Arial", 10, "bold"))
        self.status_label.pack()

    def setup_visualization_tabs(self, parent):
        """Setup enhanced visualization with multiple tabs"""
        viz_notebook = ttk.Notebook(parent)
        viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Game view tab
        game_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(game_tab, text="🎮 Game View")

        game_frame = ttk.LabelFrame(game_tab, text="🗺️ Environment Visualization", padding="10")
        game_frame.pack(fill=tk.BOTH, expand=True)

        # Game visualization
        self.game_fig, self.game_ax = plt.subplots(figsize=(10, 10), facecolor='white')
        self.game_canvas = FigureCanvasTkAgg(self.game_fig, game_frame)
        self.game_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Analytics tab
        analytics_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(analytics_tab, text="📈 Live Analytics")

        analytics_frame = ttk.LabelFrame(analytics_tab, text="📊 Training Analytics", padding="10")
        analytics_frame.pack(fill=tk.BOTH, expand=True)

        # Enhanced analytics with 2x3 subplot grid
        self.analytics_fig, self.analytics_axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_fig, analytics_frame)
        self.analytics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Performance comparison tab
        comparison_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(comparison_tab, text="📊 Performance")

        perf_frame = ttk.LabelFrame(comparison_tab, text="🏆 Performance Metrics", padding="10")
        perf_frame.pack(fill=tk.BOTH, expand=True)

        self.performance_fig, self.performance_axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, perf_frame)
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind tab change event to ensure continuous updates
        viz_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def setup_plots(self):
        """Initialize all plotting areas with enhanced styling"""
        # Game visualization
        self.game_ax.set_title("GridWorld Environment - Ready to Start!", fontsize=14, fontweight='bold')
        self.game_ax.set_xticks([])
        self.game_ax.set_yticks([])
        self.game_ax.text(0.5, 0.5, '🎮 Create an environment to begin!\n\n🔧 Configure parameters on the left\n🚀 Click "Create Environment"', 
                         transform=self.game_ax.transAxes, ha='center', va='center',
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

        # Analytics plots - enhanced layout
        self.setup_analytics_plots()
        self.setup_performance_plots()

        # Draw initial plots
        self.game_canvas.draw()
        self.analytics_canvas.draw()
        self.performance_canvas.draw()

    def setup_analytics_plots(self):
        """Setup enhanced analytics plots"""
        axes = self.analytics_axes.flatten()

        titles = [
            "📈 Episode Rewards", "📏 Episode Lengths", "🎯 Success Rate",
            "🧠 Learning Curve", "🔍 Exploration", "⚡ Training Speed"
        ]

        for i, (ax, title) in enumerate(zip(axes, titles)):
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if i < 3:
                ax.set_xlabel("Episode")

        axes[0].set_ylabel("Reward")
        axes[1].set_ylabel("Steps")
        axes[2].set_ylabel("Success Rate (%)")
        axes[3].set_ylabel("Mean Reward")
        axes[4].set_ylabel("Exploration %")
        axes[5].set_ylabel("Episodes/min")

        self.analytics_fig.tight_layout()

    def setup_performance_plots(self):
        """Setup performance comparison plots"""
        axes = self.performance_axes.flatten()

        titles = [
            "🏆 Best vs Average Performance", "📊 Reward Distribution", 
            "⏱️ Efficiency Analysis", "📈 Learning Progress"
        ]

        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        self.performance_fig.tight_layout()

    def start_continuous_updates(self):
        """Start continuous visualization updates"""
        def update_loop():
            while self.visualization_active:
                try:
                    # Process any queued updates
                    while not self.update_queue.empty():
                        update_func = self.update_queue.get_nowait()
                        self.root.after_idle(update_func)

                    # Update game visualization if environment exists
                    if self.env is not None and self.realtime_viz_var.get():
                        self.root.after_idle(self.update_game_visualization)

                    time.sleep(0.1)  # Update at 10 FPS
                except Exception as e:
                    print(f"Update loop error: {e}")
                    time.sleep(1)

        self.visualization_thread = threading.Thread(target=update_loop, daemon=True)
        self.visualization_thread.start()

    def on_tab_changed(self, event):
        """Handle tab changes to ensure continuous updates"""
        # Force update when switching tabs
        if self.env is not None:
            self.root.after_idle(self.update_game_visualization)

    def create_environment(self):
        """Create enhanced environment with error handling"""
        try:
            size = int(self.size_var.get())
            density = float(self.density_var.get())
            reward_shaping = self.reward_shaping_var.get()

            if not (5 <= size <= 25):
                raise ValueError("Grid size must be between 5 and 25")
            if not (0.05 <= density <= 0.4):
                raise ValueError("Obstacle density must be between 0.05 and 0.4")

            self.env = GridWorldEnv(
                size=size, 
                obstacle_density=density, 
                reward_shaping=reward_shaping
            )
            self.env.reset()

            self.update_game_visualization()
            self.update_status("Environment created successfully! 🎯", "#28a745")

            # Enable controls
            self.train_btn.config(state="normal")
            self.play_random_btn.config(state="normal")

            # Update stats
            self.update_stats_display(f"""Environment Created! 🌍

Grid Size: {size}x{size}
Obstacle Density: {density:.2%}
Reward Shaping: {'✓' if reward_shaping else '✗'}

🤖 Agent: Blue square
🎯 Goal: Red square  
⬛ Obstacles: Black squares
👻 Visited: Light gray

Ready for training! 🚀""")

        except ValueError as e:
            messagebox.showerror("Invalid Configuration", str(e))
            self.update_status(f"Configuration error: {str(e)}", "#dc3545")
        except Exception as e:
            messagebox.showerror("Environment Error", f"Failed to create environment: {str(e)}")
            self.update_status("Environment creation failed ❌", "#dc3545")

    def update_game_visualization(self):
        """Enhanced game visualization with better error handling"""
        if self.env is None:
            return

        try:
            # Get current state
            rgb_array = self.env.render_to_array(show_visited=self.show_visited_var.get())
            info = self.env._get_info()

            # Clear and update plot
            self.game_ax.clear()

            # Display the environment
            im = self.game_ax.imshow(rgb_array, origin='upper', interpolation='nearest', 
                                   cmap=None, aspect='equal')

            # Enhanced title with more information
            title_parts = [
                f"GridWorld {self.env.size}x{self.env.size}",
                f"Episode: {info.get('episode', 0)}",
                f"Steps: {info['steps']}/{self.env.max_episode_steps}",
                f"Distance: {info['manhattan_distance']}"
            ]

            if info.get('exploration_ratio', 0) > 0:
                title_parts.append(f"Explored: {info['exploration_ratio']:.1%}")

            title = " | ".join(title_parts)
            self.game_ax.set_title(title, fontsize=11, fontweight='bold')

            # Enhanced grid and styling
            self.game_ax.set_xticks(np.arange(-0.5, self.env.size, 1), minor=True)
            self.game_ax.set_yticks(np.arange(-0.5, self.env.size, 1), minor=True)
            self.game_ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
            self.game_ax.set_xticks([])
            self.game_ax.set_yticks([])

            # Enhanced legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc='blue', label='🤖 Agent'),
                plt.Rectangle((0, 0), 1, 1, fc='red', label='🎯 Goal'),
                plt.Rectangle((0, 0), 1, 1, fc='black', label='⬛ Obstacle'),
            ]

            if self.show_visited_var.get():
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='lightgray', label='👻 Visited'))

            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='white', ec='gray', label='⬜ Empty'))

            self.game_ax.legend(handles=legend_elements, loc='center left', 
                               bbox_to_anchor=(1, 0.5), fontsize=10)

            # Adjust layout and draw
            self.game_fig.tight_layout()
            self.game_canvas.draw_idle()

        except Exception as e:
            print(f"Game visualization error: {e}")
            # Continue without crashing

    def play_random_agent(self):
        """Play with random agent - enhanced version"""
        if self.env is None:
            messagebox.showwarning("No Environment", "Please create an environment first!")
            return

        def play_episode():
            episode_data = []
            try:
                self.update_status("Random agent playing... 🎲", "#ffc107")
                self.play_random_btn.config(state="disabled")

                obs, _ = self.env.reset()
                terminated, truncated = False, False
                total_reward = 0
                step_count = 0

                while not (terminated or truncated) and step_count < self.env.max_episode_steps:
                    action = self.env.action_space.sample()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    step_count += 1

                    # Store episode data
                    episode_data.append({
                        'step': step_count,
                        'action': int(action),
                        'reward': float(reward),
                        'total_reward': float(total_reward),
                        'info': info
                    })

                    # Update visualization
                    time.sleep(self.animation_speed_var.get())

                # Final results
                success = terminated and total_reward > 50
                efficiency = "Excellent" if step_count < 20 else "Good" if step_count < 50 else "Poor"

                result_text = f"""Random Agent Results 🎲

{'🎉 SUCCESS!' if success else '❌ Failed to reach goal'}

Steps taken: {step_count}
Total reward: {total_reward:.2f}
Efficiency: {efficiency}
Exploration: {info.get('exploration_ratio', 0):.1%}

{'Goal reached!' if terminated else 'Max steps reached'}"""

                self.root.after(0, lambda: self.update_stats_display(result_text))
                self.root.after(0, lambda: self.update_status(
                    f"Random agent: {'Success' if success else 'Failed'} | Steps: {step_count} | Reward: {total_reward:.1f}", 
                    "#28a745" if success else "#dc3545"))

            except Exception as e:
                error_msg = f"Error during random play: {str(e)}"
                self.root.after(0, lambda: self.update_status(error_msg, "#dc3545"))
                print(f"Random agent error: {e}")
                traceback.print_exc()
            finally:
                self.root.after(0, lambda: self.play_random_btn.config(state="normal"))

        threading.Thread(target=play_episode, daemon=True).start()

    def start_training(self):
        """Enhanced training with multiple algorithms"""
        if self.env is None:
            messagebox.showwarning("No Environment", "Please create an environment first!")
            return

        if self.is_training:
            messagebox.showwarning("Training Active", "Training is already in progress!")
            return

        def train_model():
            try:
                self.is_training = True

                # Update UI
                self.root.after(0, lambda: self.update_status("Initializing training... 🚀", "#007bff"))
                self.root.after(0, lambda: self.train_btn.config(state="disabled"))
                self.root.after(0, lambda: self.stop_train_btn.config(state="normal"))
                self.root.after(0, lambda: self.pause_btn.config(state="normal"))
                self.root.after(0, lambda: self.progress_bar.config(mode='indeterminate'))

                # Get training parameters
                algo = self.algo_var.get()
                total_steps = int(self.steps_var.get())
                learning_rate = float(self.lr_var.get())
                batch_size = int(self.batch_size_var.get())
                gamma = float(self.gamma_var.get())
                size = int(self.size_var.get())
                density = float(self.density_var.get())

                # Clear previous stats
                self.training_stats = {
                    'episodes': [], 'mean_rewards': [], 'last_rewards': [],
                    'episode_lengths': [], 'success_rates': [], 'best_rewards': []
                }

                # Create training environment
                env_kwargs = {
                    'size': size,
                    'obstacle_density': density,
                    'reward_shaping': self.reward_shaping_var.get()
                }

                # Algorithm-specific setup
                if algo in ["PPO", "A2C"]:
                    train_env = make_vec_env(lambda: GridWorldEnv(**env_kwargs), n_envs=1)

                    if algo == "PPO":
                        self.model = PPO(
                            "MultiInputPolicy", train_env, verbose=1,
                            learning_rate=learning_rate, batch_size=batch_size,
                            gamma=gamma, n_steps=min(2048, total_steps//10),
                            n_epochs=10, clip_range=0.2, gae_lambda=0.95
                        )
                    else:  # A2C
                        self.model = A2C(
                            "MultiInputPolicy", train_env, verbose=1,
                            learning_rate=learning_rate, gamma=gamma,
                            n_steps=min(256, total_steps//20)
                        )

                elif algo in ["DQN"]:
                    def make_dqn_env():
                        env = GridWorldEnv(**env_kwargs)
                        return FlattenDictWrapper(env)

                    train_env = make_vec_env(make_dqn_env, n_envs=1)
                    self.model = DQN(
                        "MlpPolicy", train_env, verbose=1,
                        learning_rate=learning_rate, batch_size=batch_size,
                        gamma=gamma, buffer_size=min(100000, total_steps),
                        learning_starts=max(1000, total_steps//50),
                        target_update_interval=max(500, total_steps//100),
                        exploration_fraction=0.3, exploration_final_eps=0.05
                    )

                elif algo in ["SAC", "TD3"]:
                    def make_continuous_env():
                        env = GridWorldEnv(**env_kwargs)
                        return FlattenDictWrapper(env)

                    train_env = make_vec_env(make_continuous_env, n_envs=1)

                    # Add action noise for TD3
                    if algo == "TD3":
                        n_actions = train_env.action_space.shape[-1]
                        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

                        self.model = TD3(
                            "MlpPolicy", train_env, verbose=1,
                            learning_rate=learning_rate, batch_size=batch_size,
                            gamma=gamma, action_noise=action_noise,
                            buffer_size=min(100000, total_steps)
                        )
                    else:  # SAC
                        self.model = SAC(
                            "MlpPolicy", train_env, verbose=1,
                            learning_rate=learning_rate, batch_size=batch_size,
                            gamma=gamma, buffer_size=min(100000, total_steps)
                        )

                self.root.after(0, lambda: self.update_status(f"Training {algo} model... 🧠", "#007bff"))

                # Create enhanced callback
                callback = EnhancedTrainingCallback(self, log_interval=max(1, total_steps//1000))

                # Start training
                self.model.learn(total_timesteps=total_steps, callback=callback)

                if self.is_training:  # Training completed successfully
                    self.root.after(0, lambda: self.update_status("Training completed! ✅", "#28a745"))
                    self.root.after(0, lambda: self.play_trained_btn.config(state="normal"))
                    self.root.after(0, lambda: self.save_model_btn.config(state="normal"))

                    # Auto-save model
                    model_name = f"{algo.lower()}_gridworld_{size}x{size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    self.model.save(model_name)

                    completion_stats = f"""Training Completed! 🎉

Algorithm: {algo}
Total Steps: {total_steps:,}
Environment: {size}x{size}
Success Rate: {self.training_stats['success_rates'][-1] if self.training_stats['success_rates'] else 0:.1f}%

Model saved: {model_name}

Ready to test! 🧠"""

                    self.root.after(0, lambda: self.update_stats_display(completion_stats))

            except ValueError as e:
                error_msg = f"Invalid parameters: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Training Error", error_msg))
                self.root.after(0, lambda: self.update_status(f"Training failed: {str(e)}", "#dc3545"))
            except Exception as e:
                if self.is_training:  # Only show error if not intentionally stopped
                    error_msg = f"Training error: {str(e)}"
                    self.root.after(0, lambda: messagebox.showerror("Training Error", error_msg))
                    self.root.after(0, lambda: self.update_status("Training failed ❌", "#dc3545"))
                    print(f"Training error: {e}")
                    traceback.print_exc()
            finally:
                self.is_training = False
                self.root.after(0, lambda: self.train_btn.config(state="normal"))
                self.root.after(0, lambda: self.stop_train_btn.config(state="disabled"))
                self.root.after(0, lambda: self.pause_btn.config(state="disabled"))
                self.root.after(0, lambda: self.progress_bar.config(mode='determinate', value=100))

        self.training_thread = threading.Thread(target=train_model, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Stop training safely"""
        self.is_training = False
        self.update_status("Training stopped by user ⏹️", "#ffc107")
        self.train_btn.config(state="normal")
        self.stop_train_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")

    def pause_training(self):
        """Pause/resume training (placeholder for future implementation)"""
        messagebox.showinfo("Feature Coming Soon", "Pause/Resume functionality will be available in the next version!")

    def play_trained_model(self):
        """Play with trained model - enhanced with proper error handling"""
        if self.model is None:
            messagebox.showwarning("No Model", "Please train a model first!")
            return

        def play_with_model():
            try:
                self.update_status("Trained agent playing... 🧠", "#007bff")
                self.play_trained_btn.config(state="disabled")

                obs, _ = self.env.reset()
                terminated, truncated = False, False
                total_reward = 0
                step_count = 0
                actions_taken = []

                while not (terminated or truncated) and step_count < self.env.max_episode_steps:
                    # Predict action based on algorithm
                    algo = self.algo_var.get()
                    try:
                        if algo in ["PPO", "A2C"]:
                            action, _ = self.model.predict(obs, deterministic=True)
                        elif algo == "DQN":
                            flat_obs = self._flatten_observation(obs)
                            action, _ = self.model.predict(flat_obs, deterministic=True)
                        elif algo in ["SAC", "TD3"]:
                            flat_obs = self._flatten_observation(obs)
                            action, _ = self.model.predict(flat_obs, deterministic=True)
                            # For discrete action space, convert continuous to discrete
                            action = np.argmax(action) if hasattr(action, '__len__') else int(action)

                        actions_taken.append(int(action))

                    except Exception as predict_error:
                        print(f"Prediction error: {predict_error}")
                        action = self.env.action_space.sample()  # Fallback to random

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    step_count += 1

                    time.sleep(self.animation_speed_var.get())

                # Analyze performance
                success = terminated and total_reward > 50
                optimal_steps = info.get('optimal_steps', step_count)
                efficiency_ratio = optimal_steps / step_count if step_count > 0 else 0

                if efficiency_ratio > 0.8:
                    performance = "Excellent! 🌟"
                elif efficiency_ratio > 0.6:
                    performance = "Very Good! 👍"
                elif efficiency_ratio > 0.4:
                    performance = "Good 👌"
                else:
                    performance = "Needs Improvement 📈"

                result_text = f"""Trained Agent Results 🧠

{'🎉 SUCCESS!' if success else '❌ Failed to reach goal'}

Algorithm: {algo}
Steps taken: {step_count}
Optimal steps: {optimal_steps}
Efficiency: {efficiency_ratio:.1%}
Total reward: {total_reward:.2f}
Performance: {performance}

Action distribution:
{self._analyze_actions(actions_taken)}"""

                self.root.after(0, lambda: self.update_stats_display(result_text))
                self.root.after(0, lambda: self.update_status(
                    f"Trained agent: {'Success' if success else 'Failed'} | Steps: {step_count} | Efficiency: {efficiency_ratio:.1%}", 
                    "#28a745" if success else "#dc3545"))

            except Exception as e:
                error_msg = f"Error during trained play: {str(e)}"
                print(f"Trained model error: {e}")
                traceback.print_exc()
                self.root.after(0, lambda: self.update_status(error_msg, "#dc3545"))
            finally:
                self.root.after(0, lambda: self.play_trained_btn.config(state="normal"))

        threading.Thread(target=play_with_model, daemon=True).start()

    def _flatten_observation(self, obs):
        """Safely flatten Dict observation"""
        try:
            flat_obs = []
            key_order = ['agent', 'goal', 'obstacles', 'visited', 'distance_to_goal']

            for key in key_order:
                if key in obs:
                    val = obs[key]
                    if val.ndim == 0:
                        flat_obs.append(float(val))
                    else:
                        flat_obs.extend(val.flatten().astype(float))

            return np.array(flat_obs, dtype=np.float32)
        except Exception as e:
            print(f"Flattening error: {e}")
            # Fallback: simple concatenation
            return np.concatenate([
                obs['agent'].astype(np.float32),
                obs['goal'].astype(np.float32), 
                obs['obstacles'].flatten().astype(np.float32)
            ])

    def _analyze_actions(self, actions):
        """Analyze action distribution"""
        if not actions:
            return "No actions recorded"

        action_names = ["Up", "Right", "Down", "Left"]
        action_counts = {name: 0 for name in action_names}

        for action in actions:
            if 0 <= action < len(action_names):
                action_counts[action_names[action]] += 1

        total = len(actions)
        analysis = []
        for name, count in action_counts.items():
            if count > 0:
                percentage = (count / total) * 100
                analysis.append(f"{name}: {count} ({percentage:.1f}%)")

        return "\n".join(analysis) if analysis else "No valid actions"

    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            messagebox.showwarning("No Model", "No trained model to save!")
            return

        try:
            filename = filedialog.asksaveasfilename(
                title="Save Model",
                defaultextension=".zip",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
            )

            if filename:
                self.model.save(filename)
                messagebox.showinfo("Model Saved", f"Model saved successfully as:\n{filename}")
                self.update_status(f"Model saved: {os.path.basename(filename)} 💾", "#28a745")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        """Load a pre-trained model"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Model",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
            )

            if filename:
                # Determine algorithm from filename or ask user
                algo = self.algo_var.get()

                if algo == "PPO":
                    self.model = PPO.load(filename)
                elif algo == "A2C":
                    self.model = A2C.load(filename)
                elif algo == "DQN":
                    self.model = DQN.load(filename)
                elif algo == "SAC":
                    self.model = SAC.load(filename)
                elif algo == "TD3":
                    self.model = TD3.load(filename)

                self.play_trained_btn.config(state="normal")
                self.save_model_btn.config(state="normal")

                messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{filename}")
                self.update_status(f"Model loaded: {os.path.basename(filename)} 📂", "#28a745")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")

    def save_analytics_to_json(self, analytics_data):
        """Save analytics data to JSON file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_analytics_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)

            self.update_status(f"Analytics saved: {filename} 📊", "#28a745")
            print(f"Analytics data saved to {filename}")

        except Exception as e:
            print(f"Error saving analytics: {e}")

    def export_analytics_data(self):
        """Export current analytics data manually"""
        if not self.training_stats['episodes']:
            messagebox.showwarning("No Data", "No training data to export!")
            return

        try:
            filename = filedialog.asksaveasfilename(
                title="Export Analytics Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                export_data = {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'environment_config': {
                        'grid_size': int(self.size_var.get()),
                        'obstacle_density': float(self.density_var.get()),
                        'reward_shaping': self.reward_shaping_var.get()
                    },
                    'training_config': {
                        'algorithm': self.algo_var.get(),
                        'learning_rate': float(self.lr_var.get()),
                        'batch_size': int(self.batch_size_var.get()),
                        'gamma': float(self.gamma_var.get())
                    },
                    'training_stats': self.training_stats
                }

                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

                messagebox.showinfo("Data Exported", f"Analytics data exported to:\n{filename}")
                self.update_status(f"Data exported: {os.path.basename(filename)} 📊", "#28a745")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def update_training_stats(self, episode, mean_reward, last_reward, episode_length, success_rate, best_reward):
        """Update training statistics with enhanced data"""

        # Store stats
        self.training_stats['episodes'].append(episode)
        self.training_stats['mean_rewards'].append(mean_reward)
        self.training_stats['last_rewards'].append(last_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['success_rates'].append(success_rate)
        self.training_stats['best_rewards'].append(best_reward)

        # Update progress bar
        if hasattr(self, 'steps_var'):
            try:
                total_steps = int(self.steps_var.get())
                progress = min(100, (episode * 100) / max(1, total_steps // 200))
                self.progress_var.set(progress)
            except:
                pass

        # Queue plot updates
        self.update_queue.put(self._update_analytics_plots)
        self.update_queue.put(self._update_performance_plots)

        # Update statistics display
        stats_text = f"""Training Progress 📈

Episode: {episode}
Last Reward: {last_reward:.2f}
Mean Reward: {mean_reward:.2f}
Best Reward: {best_reward:.2f}
Success Rate: {success_rate:.1f}%
Episode Length: {episode_length}

Performance Level:
{self._get_performance_emoji(success_rate, mean_reward)}

Algorithm: {self.algo_var.get()}
Environment: {self.size_var.get()}x{self.size_var.get()}

{self._get_training_advice(success_rate, mean_reward)}"""

        self.update_queue.put(lambda: self.update_stats_display(stats_text))

    def _get_performance_emoji(self, success_rate, mean_reward):
        """Get performance emoji and description"""
        if success_rate > 80 and mean_reward > 50:
            return "🌟 Excellent Performance!"
        elif success_rate > 60 and mean_reward > 30:
            return "👍 Very Good Progress!"
        elif success_rate > 40 and mean_reward > 10:
            return "📈 Good Learning Curve!"
        elif success_rate > 20 and mean_reward > 0:
            return "🎯 Making Progress!"
        else:
            return "🚀 Early Training Phase"

    def _get_training_advice(self, success_rate, mean_reward):
        """Get training advice based on performance"""
        if success_rate < 10 and mean_reward < 0:
            return "💡 Tip: Agent is still exploring. Be patient!"
        elif success_rate < 30:
            return "💡 Tip: Try adjusting learning rate or reward shaping."
        elif success_rate > 70:
            return "🎉 Great! Agent is performing well!"
        else:
            return "📊 Training is progressing nicely!"

    def _update_analytics_plots(self):
        """Update enhanced analytics plots"""
        if not self.training_stats['episodes']:
            return

        try:
            episodes = self.training_stats['episodes']
            mean_rewards = self.training_stats['mean_rewards']
            last_rewards = self.training_stats['last_rewards']
            episode_lengths = self.training_stats['episode_lengths']
            success_rates = self.training_stats['success_rates']
            best_rewards = self.training_stats['best_rewards']

            axes = self.analytics_axes.flatten()

            # Clear all plots
            for ax in axes:
                ax.clear()

            # Plot 1: Episode Rewards
            axes[0].plot(episodes, last_rewards, 'lightcoral', alpha=0.6, linewidth=1, label='Episode Reward')
            axes[0].plot(episodes, mean_rewards, 'darkred', linewidth=2, label='Moving Average')
            axes[0].fill_between(episodes, last_rewards, alpha=0.3, color='lightcoral')
            axes[0].set_title("📈 Episode Rewards", fontsize=11, fontweight='bold')
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Reward")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Episode Lengths
            axes[1].plot(episodes, episode_lengths, 'skyblue', linewidth=2, label='Steps per Episode')
            axes[1].axhline(y=np.mean(episode_lengths), color='darkblue', 
                           linestyle='--', alpha=0.7, label=f'Average: {np.mean(episode_lengths):.1f}')
            axes[1].set_title("📏 Episode Lengths", fontsize=11, fontweight='bold')
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Steps")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Success Rate
            axes[2].plot(episodes, success_rates, 'green', linewidth=2, marker='o', markersize=3)
            axes[2].fill_between(episodes, success_rates, alpha=0.3, color='green')
            axes[2].set_title("🎯 Success Rate", fontsize=11, fontweight='bold')
            axes[2].set_xlabel("Episode")
            axes[2].set_ylabel("Success Rate (%)")
            axes[2].set_ylim(0, 100)
            axes[2].grid(True, alpha=0.3)

            # Plot 4: Learning Curve  
            axes[3].plot(episodes, mean_rewards, 'purple', linewidth=2, label='Mean Reward')
            axes[3].plot(episodes, best_rewards, 'gold', linewidth=2, label='Best Reward')
            axes[3].fill_between(episodes, mean_rewards, alpha=0.3, color='purple')
            axes[3].set_title("🧠 Learning Curve", fontsize=11, fontweight='bold')
            axes[3].set_xlabel("Episode")
            axes[3].set_ylabel("Reward")
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

            # Plot 5: Exploration (based on episode lengths - shorter = more efficient)
            exploration_metric = [100 - min(100, (length / 50) * 100) for length in episode_lengths]
            axes[4].plot(episodes, exploration_metric, 'orange', linewidth=2)
            axes[4].set_title("🔍 Efficiency", fontsize=11, fontweight='bold')
            axes[4].set_xlabel("Episode")
            axes[4].set_ylabel("Efficiency %")
            axes[4].set_ylim(0, 100)
            axes[4].grid(True, alpha=0.3)

            # Plot 6: Training Speed (episodes per time window)
            if len(episodes) > 10:
                speed_metric = []
                window_size = min(10, len(episodes))
                for i in range(len(episodes)):
                    if i >= window_size - 1:
                        speed = window_size  # Episodes per window
                        speed_metric.append(speed * 6)  # Scale for visualization
                    else:
                        speed_metric.append(0)

                axes[5].plot(episodes, speed_metric, 'teal', linewidth=2)
                axes[5].set_title("⚡ Training Speed", fontsize=11, fontweight='bold')
                axes[5].set_xlabel("Episode")
                axes[5].set_ylabel("Episodes/min (est.)")
                axes[5].grid(True, alpha=0.3)

            # Adjust layout and draw
            self.analytics_fig.tight_layout()
            self.analytics_canvas.draw_idle()

        except Exception as e:
            print(f"Analytics plot error: {e}")

    def _update_performance_plots(self):
        """Update performance comparison plots"""
        if not self.training_stats['episodes']:
            return

        try:
            episodes = self.training_stats['episodes']
            mean_rewards = self.training_stats['mean_rewards']
            last_rewards = self.training_stats['last_rewards']
            best_rewards = self.training_stats['best_rewards']
            success_rates = self.training_stats['success_rates']

            axes = self.performance_axes.flatten()

            # Clear plots
            for ax in axes:
                ax.clear()

            # Plot 1: Best vs Average Performance
            axes[0].plot(episodes, best_rewards, 'gold', linewidth=3, label='Best Reward', marker='*', markersize=4)
            axes[0].plot(episodes, mean_rewards, 'blue', linewidth=2, label='Average Reward')
            axes[0].fill_between(episodes, mean_rewards, best_rewards, alpha=0.2, color='gold')
            axes[0].set_title("🏆 Best vs Average Performance", fontsize=11, fontweight='bold')
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Reward")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Reward Distribution
            if len(last_rewards) > 5:
                axes[1].hist(last_rewards, bins=min(20, len(last_rewards)//2), alpha=0.7, color='skyblue', edgecolor='black')
                axes[1].axvline(x=np.mean(last_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(last_rewards):.1f}')
                axes[1].set_title("📊 Reward Distribution", fontsize=11, fontweight='bold')
                axes[1].set_xlabel("Reward")
                axes[1].set_ylabel("Frequency")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            # Plot 3: Efficiency Analysis
            if len(episodes) > 1:
                efficiency = [(r / max(1, l)) * 10 for r, l in zip(last_rewards, self.training_stats['episode_lengths'])]
                axes[2].plot(episodes, efficiency, 'green', linewidth=2, marker='o', markersize=3)
                axes[2].set_title("⏱️ Efficiency Analysis", fontsize=11, fontweight='bold')
                axes[2].set_xlabel("Episode")
                axes[2].set_ylabel("Reward/Step Ratio")
                axes[2].grid(True, alpha=0.3)

            # Plot 4: Learning Progress (cumulative success)
            if len(success_rates) > 0:
                cumulative_success = np.cumsum([1 if sr > 50 else 0 for sr in success_rates])
                axes[3].plot(episodes, cumulative_success, 'purple', linewidth=3, marker='s', markersize=3)
                axes[3].fill_between(episodes, cumulative_success, alpha=0.3, color='purple')
                axes[3].set_title("📈 Cumulative Successes", fontsize=11, fontweight='bold')
                axes[3].set_xlabel("Episode")
                axes[3].set_ylabel("Total Successes")
                axes[3].grid(True, alpha=0.3)

            # Adjust layout and draw
            self.performance_fig.tight_layout()
            self.performance_canvas.draw_idle()

        except Exception as e:
            print(f"Performance plot error: {e}")

    def update_status(self, message, color="#000000"):
        """Update status with color coding"""
        self.status_label.config(text=f"Status: {message}", foreground=color)

    def update_stats_display(self, text):
        """Update statistics display safely"""
        try:
            self.stats_text.config(state='normal')
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, text)
            self.stats_text.config(state='disabled')
            self.stats_text.see(tk.END)
        except Exception as e:
            print(f"Stats display error: {e}")

    def on_closing(self):
        """Handle application closing safely"""
        if self.is_training:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit and lose progress?"):
                self.is_training = False
                self.visualization_active = False
                time.sleep(0.5)  # Give threads time to stop
                self.root.destroy()
        else:
            self.visualization_active = False
            time.sleep(0.2)  # Give threads time to stop
            self.root.destroy()

def main():
    """
    Main function to start the Enhanced RL Game Application
    """

    print("🎮 Starting Enhanced RL Game - Professional Edition")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return

    # Check required packages
    required_packages = [
        ('tkinter', 'GUI framework'), 
        ('numpy', 'numerical computing'),
        ('matplotlib', 'plotting'),
        ('gymnasium', 'RL environments'),
        ('stable_baselines3', 'RL algorithms')
    ]

    missing_packages = []
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - {description} (MISSING)")

    if missing_packages:
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return

    try:
        # Create and start the application
        root = tk.Tk()
        app = EnhancedRLGameApp(root)

        print("\n🚀 Application launched successfully!")
        print("📖 Features:")
        print("   • Multiple RL algorithms (PPO, A2C, DQN, SAC, TD3)")
        print("   • Real-time visualization and analytics")
        print("   • Advanced training controls and monitoring")
        print("   • JSON data export and model saving")
        print("   • Enhanced error handling and stability")
        print("=" * 60)
        print("🎯 Ready to train! Create an environment to begin.")

        # Start the GUI event loop
        root.mainloop()

    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
