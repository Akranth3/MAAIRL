# config.py
import torch
from pettingzoo.mpe import simple_adversary_v3
from matplotlib import pyplot as plt
import pickle
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="QMIX Training")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    return parser.parse_args()

args = get_args()

class Config:
    def __init__(self):
        # Environment settings
        self.n_agents = 3
        self.state_dim = [8, 10, 10]  # Individual agent observation dimensions
        self.action_dim = 5
        self.state_shape = sum(self.state_dim)  # Global state dimension
        
        # Training hyperparameters
        self.lr = 1e-4
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_size = 5000
        self.target_update = 200  # Update target network every n steps
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.hidden_dim = 128
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MixingNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim):
        super(MixingNetwork, self).__init__()
        
        # Hypernetwork layers
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # State-dependent bias
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_qs, states):
        # First layer weights and bias
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.n_agents, self.hidden_dim)
        
        # Second layer weights and bias
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        
        # Forward pass
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1)
        y = torch.bmm(hidden.unsqueeze(1), w2.unsqueeze(2)).squeeze() + b2
        
        return y

# memory.py
from collections import namedtuple, deque
import random

Experience = namedtuple('Experience', 
    ('state', 'action', 'reward', 'next_state', 'done', 'global_state', 'next_global_state'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# agent.py
class QMIXAgent:
    def __init__(self, config):
        self.config = config
        
        # Create Q-networks for each agent
        self.q_networks = [QNetwork(config.state_dim[i], config.action_dim, config.hidden_dim).to(config.device) 
                          for i in range(config.n_agents)]
        self.target_q_networks = [QNetwork(config.state_dim[i], config.action_dim, config.hidden_dim).to(config.device) 
                                 for i in range(config.n_agents)]
        
        # Create mixing network
        self.mixer = MixingNetwork(config.state_shape, config.n_agents, config.hidden_dim).to(config.device)
        self.target_mixer = MixingNetwork(config.state_shape, config.n_agents, config.hidden_dim).to(config.device)
        
        # Copy parameters to target networks
        self.update_target_networks()
        
        # Initialize optimizers
        self.q_params = list(self.q_networks[0].parameters())
        for i in range(1, config.n_agents):
            self.q_params += list(self.q_networks[i].parameters())
        self.mixer_params = list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.q_params + self.mixer_params, lr=config.lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(config.buffer_size)
        
        # Initialize epsilon for exploration
        self.epsilon = config.epsilon_start
        
    def select_actions(self, observations, training=True):
        actions = []
        
        for i, obs in enumerate(observations):
            if training and random.random() < self.epsilon:
                action = random.randrange(self.config.action_dim)
            else:
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    q_values = self.q_networks[i](state)
                    action = q_values.max(1)[1].item()
            actions.append(action)
            
        return actions
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.config.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.config.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.config.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.config.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.config.device)
        global_states = torch.FloatTensor([e.global_state for e in batch]).to(self.config.device)
        next_global_states = torch.FloatTensor([e.next_global_state for e in batch]).to(self.config.device)
        
        # Calculate current Q-values
        current_q_values = [self.q_networks[i](states[:,i]) for i in range(self.config.n_agents)]
        current_q_taken = torch.stack([q_values.gather(1, actions[:,i].unsqueeze(1)) 
                                     for i, q_values in enumerate(current_q_values)], dim=1).squeeze(-1)
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = [self.target_q_networks[i](next_states[:,i]) for i in range(self.config.n_agents)]
            next_q_max = torch.stack([q_values.max(1)[0] for q_values in next_q_values], dim=1)
            
            # Mix next Q-values
            target_mixed_q = self.target_mixer(next_q_max, next_global_states)
            targets = rewards + (1 - dones) * self.config.gamma * target_mixed_q
        
        # Mix current Q-values
        mixed_q = self.mixer(current_q_taken, global_states)
        
        # Calculate loss and update networks
        loss = F.mse_loss(mixed_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                         self.epsilon * self.config.epsilon_decay)
    
    def update_target_networks(self):
        for i in range(self.config.n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

# main.py
def train():
    config = Config()
    env = simple_adversary_v3.env()
    agent = QMIXAgent(config)
    
    agent_to_idx = {"adversary_0": 0, "agent_0": 1, "agent_1": 2}
    plot_rewards = []
    log_rewards = []
    step_count = 0
    
    for episode in range(args.episodes):
        env.reset()
        episode_reward = 0
        agent_wise_rewards = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
        
        # Initialize episode variables
        observations = []
        actions = []
        rewards = []
        
        # Main episode loop
        for agent_name in env.agent_iter():
            # Get current observation and reward
            observation, reward, done, truncated, info = env.last()
            
            # Store observation
            observations.append(observation)
            
            if done or truncated:
                break
                
            # Select and perform action
            if len(observations) == config.n_agents:
                actions = agent.select_actions(observations)
                observations = []
            
            current_action = actions[agent_to_idx[agent_name]]
            env.step(current_action)
            
            # Update rewards
            episode_reward += reward
            agent_wise_rewards[agent_name] += reward
            
            # Store experience in replay buffer
            if len(observations) == config.n_agents:
                global_state = np.concatenate(observations)
                next_observations = [env.observe(a) for a in env.agents]
                next_global_state = np.concatenate(next_observations)
                
                for i, agent_name in enumerate(env.agents):
                    agent.memory.push(
                        observations[i],
                        actions[i],
                        rewards[i],
                        next_observations[i],
                        done,
                        global_state,
                        next_global_state
                    )
                
                observations = []
                rewards = []
            
            # Update networks
            step_count += 1
            if step_count % config.batch_size == 0:
                agent.update(config.batch_size)
            
            # Update target networks
            if step_count % config.target_update == 0:
                agent.update_target_networks()
        
        # Store episode results
        plot_rewards.append(episode_reward)
        log_rewards.append(agent_wise_rewards)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")
    
    # Save results
    plt.figure(figsize=(15,5))
    plt.plot(plot_rewards)
    plt.savefig("rewards.png")
    pickle.dump(plot_rewards, open("plot_rewards.pkl", "wb"))
    
    return agent

if __name__ == "__main__":
    agent = train()
    # Save model parameters
    for i in range(3):
        torch.save(agent.q_networks[i].state_dict(), f"check_points/q_network_agent_{i}.pth")
    torch.save(agent.mixer.state_dict(), "check_points/mixer.pth")