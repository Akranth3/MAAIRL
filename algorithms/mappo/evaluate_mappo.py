# code to evaluate the mappo agent.
import torch
from mappo import MAPPOAgent
from pettingzoo.mpe import simple_adversary_v3
import numpy as np

class Config:
    def __init__(self):
        # Environment settings
        self.n_agents = 3
        self.state_dim = [8, 10, 10]
        self.action_dim = 5
        
        # Training hyperparameters
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.update_timestep = 2000
        self.hidden_dim = 128
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
env = simple_adversary_v3.env(max_cycles=100, render_mode="rgb_array")  # Make sure you have the environment installed

agents = [MAPPOAgent(config.state_dim[i], config.action_dim, config) 
            for i in range(config.n_agents)]

for i in range(3):
    agents[i].actor.load_state_dict(torch.load(f"check_points/actor_agent_{i}.pth"))

# see actor netorks
for i in range(3):
    print(agents[i].actor)

print("actors loaded successfully")

# evaluate the agents
num_episodes = 5
agent_to_idx = {"adversary_0":0, "agent_0":1, "agent_1":2}

for episode in range(num_episodes):

    env.reset()
    image_arrays = []
    episodic_reward = 0
    for agent in env.agent_iter():
        observation, reward, done, truncated,_ = env.last()

        if done or truncated:
            action = None
        else:
            action, prob = agents[agent_to_idx[agent]].select_action(observation)
        # print("action: ", action)
        env.step(action)
        episodic_reward += reward
        rgb_array = env.render()
        image_arrays.append(rgb_array)

    image_arrays = np.array(image_arrays)
    print(image_arrays.shape)
    np.save(f"videos/episode_{episode}.npy", image_arrays)

    
    print(f"Episode {episode} reward: {episodic_reward}")
