# main.py
#
# A complete application for training a Stable Baselines3 reinforcement learning agent
# with a real-time analytics graph and a user interface built with Tkinter.
#
# --- SETUP ---
# 1. Ensure you have Python 3.10+ installed.
# 2. It's recommended to use a virtual environment.
# 3. Install PyTorch first: https://pytorch.org/get-started/locally/
# 4. Install other dependencies: pip install stable-baselines3[extra] gymnasium matplotlib

import tkinter as tk
from tkinter import ttk
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import os
import time

class PlottingCallback(BaseCallback):
    """
    A custom callback that logs the mean reward of each episode
    and puts it into a queue for the main GUI thread to process.
    """
    def __init__(self, reward_queue: queue.Queue, verbose: int = 0):
        super(PlottingCallback, self).__init__(verbose)
        self.reward_queue = reward_queue

    def _on_step(self) -> bool:
        # The Monitor wrapper logs episode statistics in the 'info' dictionary.
        # We check for this dictionary to get the reward of a completed episode.
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                episode_reward = info['episode']['r']
                # Put the current timestep and the episode reward into the queue
                self.reward_queue.put((self.num_timesteps, episode_reward))
        return True

class RLTrainerApp:
    """
    The main application class that sets up the GUI and handles the
    reinforcement learning logic.
    """
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("RL Agent Trainer - CartPole")
        master.geometry("800x600")

        # --- Configuration ---
        self.model_path = "ppo_cartpole.zip"
        self.log_dir = os.path.join(os.getcwd(), "tmp", "gym")
        os.makedirs(self.log_dir, exist_ok=True)
        self.total_timesteps = 25000

        # --- State Variables ---
        self.trained_model = None
        self.is_training = False
        self.is_playing = False
        self.reward_queue = queue.Queue()

        # --- GUI Setup ---
        self.setup_gui()

        # Check if a pre-trained model exists to enable the 'Play Trained' button
        self.update_button_states()

    def setup_gui(self):
        """Initializes all the GUI widgets."""
        # --- Main Layout Frames ---
        control_frame = ttk.Frame(self.master, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        analytics_frame = ttk.Frame(self.master, padding="10")
        analytics_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        self.btn_play_untrained = ttk.Button(control_frame, text="Play with Untrained Agent", command=self.play_untrained)
        self.btn_play_untrained.pack(side=tk.LEFT, padx=5)

        self.btn_train = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.btn_train.pack(side=tk.LEFT, padx=5)

        self.btn_play_trained = ttk.Button(control_frame, text="Play with Trained Agent", command=self.play_trained)
        self.btn_play_trained.pack(side=tk.LEFT, padx=5)
        
        self.training_status_label = ttk.Label(control_frame, text="Status: Idle", font=("Helvetica", 10))
        self.training_status_label.pack(side=tk.LEFT, padx=10, pady=5)

        # --- Matplotlib Graph ---
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Mean Reward per Episode")
        self.line, = self.ax.plot([], [], 'r-') # Start with empty data
        self.ax.grid()

        self.canvas = FigureCanvasTkAgg(self.fig, master=analytics_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.timesteps_data = []
        self.rewards_data = []

    def update_plot(self):
        """
        Periodically checks the reward queue for new data and updates the plot.
        This function is run on the main GUI thread.
        """
        if self.is_training:
            try:
                while not self.reward_queue.empty():
                    timestep, reward = self.reward_queue.get_nowait()
                    self.timesteps_data.append(timestep)
                    self.rewards_data.append(reward)
                
                if self.timesteps_data:
                    self.line.set_data(self.timesteps_data, self.rewards_data)
                    self.ax.relim()
                    self.ax.autoscale_view(True, True, True)
                    self.fig.canvas.draw()

            except queue.Empty:
                pass
            finally:
                # Reschedule the next update
                self.master.after(500, self.update_plot)

    def update_button_states(self):
        """Enables or disables buttons based on the application's state."""
        is_model_available = os.path.exists(self.model_path)

        if self.is_training:
            self.btn_play_untrained.config(state=tk.DISABLED)
            self.btn_train.config(state=tk.DISABLED)
            self.btn_play_trained.config(state=tk.DISABLED)
        elif self.is_playing:
            self.btn_play_untrained.config(state=tk.DISABLED)
            self.btn_train.config(state=tk.DISABLED)
            self.btn_play_trained.config(state=tk.DISABLED)
        else: # Idle state
            self.btn_play_untrained.config(state=tk.NORMAL)
            self.btn_train.config(state=tk.NORMAL)
            self.btn_play_trained.config(state=tk.NORMAL if is_model_available else tk.DISABLED)

    def _run_game_thread(self, agent=None):
        """
        Target function for the game-playing thread.
        Handles creating the environment and running one episode.
        """
        self.is_playing = True
        self.master.after(0, self.update_button_states)
        self.master.after(0, self.training_status_label.config, {'text': f"Status: Running {'Trained' if agent else 'Untrained'} Agent..."})
        
        try:
            env = gym.make("CartPole-v1", render_mode="human")
            obs, _ = env.reset()
            done, truncated = False, False
            while not done and not truncated:
                if agent:
                    action, _ = agent.predict(obs, deterministic=True)
                else: # Untrained agent
                    action = env.action_space.sample()
                obs, _, done, truncated, _ = env.step(action)
                env.render()
                time.sleep(0.01) # Slow down rendering a bit
            env.close()
        finally:
            self.is_playing = False
            self.master.after(0, self.update_button_states)
            self.master.after(0, self.training_status_label.config, {'text': "Status: Idle"})

    def play_untrained(self):
        """Starts a thread to play with a random agent."""
        if self.is_playing or self.is_training: return
        threading.Thread(target=self._run_game_thread, args=(None,), daemon=True).start()

    def play_trained(self):
        """Loads the trained model and starts a thread to play."""
        if self.is_playing or self.is_training: return
        try:
            if self.trained_model is None:
                self.trained_model = PPO.load(self.model_path)
            threading.Thread(target=self._run_game_thread, args=(self.trained_model,), daemon=True).start()
        except FileNotFoundError:
            self.training_status_label.config(text="Status: Error - No trained model found!")
        except Exception as e:
            self.training_status_label.config(text=f"Status: Error - {e}")

    def start_training(self):
        """Initiates the model training process in a separate thread."""
        if self.is_training or self.is_playing: return
        
        self.is_training = True
        self.update_button_states()
        self.training_status_label.config(text=f"Status: Training... (0/{self.total_timesteps} steps)")

        # Clear previous training data from plot
        self.timesteps_data.clear()
        self.rewards_data.clear()
        self.line.set_data([], [])
        self.fig.canvas.draw()

        # Start training in a separate thread to not freeze the GUI
        self.training_thread = threading.Thread(target=self._train_model_thread, daemon=True)
        self.training_thread.start()
        
        # Start the plot updater loop
        self.master.after(500, self.update_plot)

    def _train_model_thread(self):
        """
        Target function for the training thread. Sets up the environment,
        model, and callback, then starts the learning process.
        """
        try:
            # The Monitor wrapper is crucial for logging episode rewards
            env = Monitor(gym.make("CartPole-v1"), self.log_dir)
            
            callback = PlottingCallback(self.reward_queue)
            
            # Removed the tensorboard_log argument to prevent error if tensorboard is not installed
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=self.total_timesteps, callback=callback)
            model.save(self.model_path)
            self.trained_model = model

        except Exception as e:
            # Handle potential errors during training
            print(f"An error occurred during training: {e}")
        finally:
            # Schedule the completion actions to be run on the main thread
            self.master.after(0, self.on_training_complete)

    def on_training_complete(self):
        """
        Called on the main thread after training is finished.
        Updates the GUI state.
        """
        self.is_training = False
        self.update_button_states()
        self.training_status_label.config(text="Status: Training Complete!")

if __name__ == "__main__":
    print("Setting up the application...")
    print("Please ensure you have the required libraries installed.")
    print("For setup instructions, see the README.md file.")
    
    root = tk.Tk()
    app = RLTrainerApp(root)
    root.mainloop()

