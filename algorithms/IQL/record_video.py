import torch

from algorithms.IQL.iql import IQLAgent
from pettingzoo.mpe import simple_adversary_v3
import numpy as np

class Config:
    def __init__(self):
        # Environment settings
        self.n_agents = 3
        self.state_dim = [8, 10, 10]  # State dimensions for each agent
        self.action_dim = 5
        
        # Training hyperparameters
        self.lr = 1e-3
        self.gamma = 0.99
        self.batch_size = 64
        self.target_update = 100  # Update target network every N steps
        self.memory_size = 100000
        self.hidden_dim = 128
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



env = simple_adversary_v3.env(render_mode="rgb_array", max_cycles=100)  # Specify render_mode here
config = Config()

# Initialize agents
agents = [IQLAgent(config.state_dim[i], config.action_dim, config) 
            for i in range(config.n_agents)]

agent_to_idx = {"adversary_0": 0, "agent_0": 1, "agent_1": 2}
# epsilon = args.epsilon
plot_rewards = []
log_rewards = []

for episode in range(2):
    env.reset()
    episode_reward = 0
    agent_wise_rewards = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
    # Main training loop
    image_arrays = []
    for agent in env.agent_iter():
        observation, reward, done, truncated, info = env.last()
        
        # Select and perform action
        action = agents[agent_to_idx[agent]].select_action(observation, epsilon=0.0)
        
        if done or truncated:
            break
            
        # Step environment
        env.step(action)
        episode_reward += reward
        rgb_array = env.render()
        # rgb_array = np.expand_dims(rgb_array, axis=0)  # Add batch dimension
        image_arrays.append(rgb_array)
        # image_arrays = np.concatenate(rgb_array, axis=0)
        # print(type(rgb_array), rgb_array.shape)
    image_arrays = np.array(image_arrays)
    dump_location = "videos/episode_{}.npy".format(episode)
    np.save(dump_location, image_arrays)
    print(image_arrays.shape)
    print(f"Episode {episode} reward: {episode_reward}")
    print(f"Episode length: {env.steps}")

# First, install required dependencies:
# pip install moviepy

# import torch
# from algorithms.IQL.iql import IQLAgent
# from pettingzoo.mpe import simple_adversary_v3
# from gymnasium.wrappers import RecordVideo
# import os
# from datetime import datetime
# try:
#     import moviepy
# except ImportError:
#     raise ImportError("Please install moviepy first: pip install moviepy")

# class Config:
#     def __init__(self):
#         # Environment settings
#         self.n_agents = 3
#         self.state_dim = [8, 10, 10]  # State dimensions for each agent
#         self.action_dim = 5
#         # Training hyperparameters
#         self.lr = 1e-3
#         self.gamma = 0.99
#         self.batch_size = 64
#         self.target_update = 100  # Update target network every N steps
#         self.memory_size = 100000
#         self.hidden_dim = 128
#         # Device configuration
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Recording settings
#         self.video_dir = os.path.join(os.getcwd(), "videos")
#         self.recording_interval = 1  # Record every N episodes

# def setup_recording(env, video_dir):
#     """Wrap the environment with RecordVideo wrapper"""
#     os.makedirs(video_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     video_folder = os.path.join(video_dir, f"run_{timestamp}")
    
#     # Define when to record videos (every N episodes)
#     def episode_trigger(episode_id):
#         return episode_id % config.recording_interval == 0
    
#     # Enable rendering for video recording
#     env.render_mode = "rgb_array"
    
#     wrapped_env = RecordVideo(
#         env,
#         video_folder=video_folder,
#         episode_trigger=episode_trigger,
#         name_prefix="episode"
#     )
#     return wrapped_env

# # Initialize configuration
# config = Config()

# # Create and wrap the environment
# env = simple_adversary_v3.env(render_mode="rgb_array")  # Specify render_mode here
# env = setup_recording(env, config.video_dir)

# # Initialize agents
# agents = [IQLAgent(config.state_dim[i], config.action_dim, config)
#           for i in range(config.n_agents)]
# agent_to_idx = {"adversary_0": 0, "agent_0": 1, "agent_1": 2}

# plot_rewards = []
# log_rewards = []

# try:
#     for episode in range(2):
#         env.reset()
#         episode_reward = 0
#         agent_wise_rewards = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
        
#         # Main training loop
#         for agent in env.agent_iter():
#             observation, reward, done, truncated, info = env.last()
            
#             # Select and perform action
#             action = agents[agent_to_idx[agent]].select_action(observation, epsilon=0.0)
            
#             if done or truncated:
#                 break
                
#             # Step environment
#             env.step(action)
#             episode_reward += reward
            
#         print(f"Episode {episode} reward: {episode_reward}")
#         print(f"Episode length: {env.steps}")
        
# except Exception as e:
#     print(f"An error occurred: {e}")
# finally:
#     # Ensure proper cleanup
#     env.close()
#     print("Environment closed properly")