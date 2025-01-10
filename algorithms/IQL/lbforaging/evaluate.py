import gymnasium as gym
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from collections import deque
import random

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate IQL agents in LBForaging")
    parser.add_argument("--episodes", type=int, default=100, help="number of evaluation episodes")
    parser.add_argument("--model_path", type=str, default="networks", help="directory containing saved models")
    parser.add_argument("--model_prefix", type=str, default="q_network_", help="prefix of model files")
    parser.add_argument("--model_suffix", type=str, default="_final_30000.pth", help="suffix of model files")
    return parser.parse_args()

class MultiAgentFCNetwork(nn.Module):
    def __init__(self, in_sizes: list[int], out_sizes: list[int], hidden_dims=(64, 64)):
        super().__init__()
        self.networks = nn.ModuleList()
        
        for in_size, out_size in zip(in_sizes, out_sizes):
            network = nn.Sequential(
                nn.Linear(in_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], out_size)
            )
            self.networks.append(network)

    def forward(self, inputs: list[torch.Tensor]):
        return [network(x) for x, network in zip(inputs, self.networks)]

def register_env(size=8, players=2, food=1, force_coop=False):
    register(
        id=f"Foraging-{size}x{size}-{players}p-{food}f{'-coop' if force_coop else ''}-v3",
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": players,
            "min_player_level": 1,
            "max_player_level": 3,
            "min_food_level": 1,
            "max_food_level": 3,
            "field_size": (size, size),
            "max_num_food": food,
            "sight": size,
            "max_episode_steps": 50,
            "force_coop": force_coop,
            "normalize_reward": True,
            "grid_observation": False,
            "observe_agent_levels": True,
            "penalty": 0.0,
            "render_mode": None
        }
    )

def select_action(state, q_network, device):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_network(state_tensor)
        return q_values.argmax(1).item()

def evaluate():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment setup
    register_env()
    env = gym.make("Foraging-8x8-2p-1f-v3")
    
    # Network initialization
    obs_sizes = [9, 9]  # Observation space size for each agent
    action_sizes = [6, 6]  # Action space size for each agent
    
    q_networks = MultiAgentFCNetwork(obs_sizes, action_sizes).to(device)
    
    # Load saved networks
    for i, network in enumerate(q_networks.networks):
        model_path = os.path.join(args.model_path, f"{args.model_prefix}{i}{args.model_suffix}")
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
    
    # Evaluation metrics
    success_count = 0
    episode_lengths = []
    episode_rewards = []
    coordination_count = 0  # Count episodes where both agents received positive rewards
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = np.zeros(len(obs_sizes))
        step_count = 0
        
        while True:
            actions = []
            for i, agent_obs in enumerate(obs):
                action = select_action(agent_obs, q_networks.networks[i], device)
                actions.append(action)
            
            next_obs, rewards, done, truncated, _ = env.step(actions)
            episode_reward += np.array(rewards)
            step_count += 1
            
            obs = next_obs
            
            if done or truncated:
                break
        
        # Record metrics
        episode_lengths.append(step_count)
        episode_rewards.append(episode_reward)
        if step_count < 50:  # Successfully completed before timeout
            success_count += 1
        if all(r > 0 for r in episode_reward):  # Both agents got positive rewards
            coordination_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"Episode Length: {step_count}")
            print(f"Episode Rewards: {episode_reward}")
            print("------------------------")
    
    # Calculate and print final statistics
    success_rate = (success_count / args.episodes) * 100
    coordination_rate = (coordination_count / args.episodes) * 100
    avg_episode_length = np.mean(episode_lengths)
    avg_rewards = np.mean(episode_rewards, axis=0)
    std_rewards = np.std(episode_rewards, axis=0)
    
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Coordination Rate: {coordination_rate:.2f}%")
    print(f"Average Episode Length: {avg_episode_length:.2f}")
    print(f"Average Rewards: Agent 1: {avg_rewards[0]:.2f}, Agent 2: {avg_rewards[1]:.2f}")
    print(f"Reward Standard Deviation: Agent 1: {std_rewards[0]:.2f}, Agent 2: {std_rewards[1]:.2f}")

if __name__ == "__main__":
    evaluate()