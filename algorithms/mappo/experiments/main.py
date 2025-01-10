# config.py
import torch
from pettingzoo.mpe import simple_adversary_v3
from matplotlib import pyplot as plt
import pickle
import argparse
import os
from itertools import product
import json
from datetime import datetime
from algorithms.mappo.mappo import MAPPOAgent

def get_args():
    parser = argparse.ArgumentParser(description="MAPPO Training")
    parser.add_argument("--episodes", type=int, default=7000, help="Number of episodes to train")
    parser.add_argument("--output_dir", type=str, default="hyperparameter_search", help="Directory to store results")
    return parser.parse_args()

class Config:
    def __init__(self, **kwargs):
        # Environment settings
        self.n_agents = 3
        self.state_dim = [8, 10, 10]
        self.action_dim = 5
        
        # Training hyperparameters - set defaults
        self.lr_actor = kwargs.get('lr_actor', 1e-4)
        self.lr_critic = kwargs.get('lr_critic', 1e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_clip = kwargs.get('eps_clip', 0.2)
        self.k_epochs = kwargs.get('k_epochs', 10)
        self.update_timestep = kwargs.get('update_timestep', 2048)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.batch_size = kwargs.get('batch_size', 64)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a unique identifier for this configuration
        self.config_id = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self):
        return {
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'k_epochs': self.k_epochs,
            'update_timestep': self.update_timestep,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size,
            'gae_lambda': self.gae_lambda,
            'entropy_coef': self.entropy_coef
        }

def generate_hyperparameter_configs():
    # Define hyperparameter search space
    hyperparameters = {
        'lr_actor': [1e-4,  1e-3],
        'lr_critic': [1e-4, 1e-3],
        'hidden_dim': [64 , 256],
        'eps_clip': [0.1, 0.2],
        'k_epochs': [5, 10],
        'update_timestep': [1024, 2048],
        'batch_size': [64],
        'gae_lambda': [0.9, 0.95],
        'gamma': [0.99],
        'entropy_coef': [0.01, 0.02]
    }
    
    # Generate all combinations
    keys, values = zip(*hyperparameters.items())
    configurations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Select first 10 configurations (you can modify this selection strategy)
    return configurations[:10]

def save_results(config, rewards, agents, output_dir):
    # Create directory for this configuration
    config_dir = os.path.join(output_dir, config.config_id)
    os.makedirs(config_dir, exist_ok=True)
    
    # Save hyperparameter configuration
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    
    # Save rewards plot
    plt.figure(figsize=(15,5))
    plt.plot(rewards)
    plt.title(f'Rewards over Episodes\nConfig: {config.config_id}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(config_dir, 'rewards_plot.png'))
    plt.close()
    
    # Save raw reward values
    with open(os.path.join(config_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(rewards, f)
    
    # Save model checkpoints
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), 
                  os.path.join(config_dir, f'actor_agent_{i}.pth'))
        torch.save(agent.critic.state_dict(), 
                  os.path.join(config_dir, f'critic_agent_{i}.pth'))

def train_single_config(config, args):
    env = simple_adversary_v3.env()
    agents = [MAPPOAgent(config.state_dim[i], config.action_dim, config) 
              for i in range(config.n_agents)]
    
    agent_to_idx = {"adversary_0":0, "agent_0":1, "agent_1":2}
    plot_rewards = []
    time_step = 0
    
    for episode in range(args.episodes):
        env.reset()
        episode_reward = 0
        done = False
        
        actions = []
        action_probs = []
        states = []
        next_states = []
        rewards = []
        
        for agent in env.agent_iter():
            observation, reward, done, truncated, info = env.last()
            action, action_prob = agents[agent_to_idx[agent]].select_action(observation)
            
            actions.append(action)
            action_probs.append(action_prob)
            states.append(observation)
            rewards.append(reward)
            
            if done or truncated:
                break
            
            env.step(action)
            
            if agent == "agent_1":
                next_states = [env.observe(a) for a in env.agents]
                for i, ag in enumerate(agents):
                    ag.memory.push(
                        states[i],
                        actions[i],
                        rewards[i],
                        next_states[i],
                        False,
                        action_probs[i]
                    )
                states = []
                next_states = []
                rewards = []
                actions = []
                action_probs = []
            
            episode_reward += reward
            time_step += 1
            
            if time_step % config.update_timestep == 0:
                for agent in agents:
                    agent.update()
        
        plot_rewards.append(episode_reward)
        if episode % 100 == 0:
            print(f"Config {config.config_id} - Episode {episode}, Reward: {episode_reward}")
    
    return agents, plot_rewards

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate configurations
    configs = generate_hyperparameter_configs()
    
    # Train and evaluate each configuration
    for config_dict in configs:
        config = Config(**config_dict)
        print(f"\nTraining configuration: {config.config_id}")
        print(json.dumps(config.to_dict(), indent=2))
        
        agents, rewards = train_single_config(config, args)
        save_results(config, rewards, agents, args.output_dir)

if __name__ == "__main__":
    main()