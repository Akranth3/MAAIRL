import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from typing import Dict, List, Tuple
import pickle
import lbforaging
import gymnasium as gym

class RewardNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, actions):
        input_tensor = torch.cat([obs, actions], dim=-1)
        return self.network(input_tensor)

class Discriminator(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.reward = RewardNetwork(obs_dim, act_dim)
        self.value = RewardNetwork(obs_dim, 0)  # State-only network for potential shaping
        
    def forward(self, obs, next_obs, actions):
        """f_w,φ(s_t, a_t, s_t+1) = g_w(s_t, a_t) + γh_φ(s_t+1) - h_φ(s_t)"""
        reward_val = self.reward(obs, actions)
        value_next = self.value(next_obs, None)
        value_current = self.value(obs, None)
        
        return reward_val + 0.99 * value_next - value_current

class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, obs):
        features = self.network(obs)
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

class MAAIRL:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_agents: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        
        # Initialize discriminators and policies for each agent
        self.discriminators = nn.ModuleList([
            Discriminator(obs_dim, act_dim) for _ in range(n_agents)
        ])
        
        self.policies = nn.ModuleList([
            Policy(obs_dim, act_dim) for _ in range(n_agents)
        ])
        
        # Initialize optimizers
        self.disc_optimizers = [
            optim.Adam(disc.parameters(), lr=lr) for disc in self.discriminators
        ]
        self.policy_optimizers = [
            optim.Adam(policy.parameters(), lr=lr) for policy in self.policies
        ]
        
    def process_trajectory_data(self, trajectories: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert trajectory dictionary to tensors"""
        all_obs = []
        all_next_obs = []
        all_actions = []
        
        for traj_idx in range(len(trajectories)):
            # Process each agent's data
            agent_1_data = trajectories[traj_idx]['agent_1']
            agent_2_data = trajectories[traj_idx]['agent_2']
            
            # Combine observations and actions
            obs = torch.tensor(np.vstack([
                agent_1_data['observations'][:-1],
                agent_2_data['observations'][:-1]
            ]), dtype=torch.float32)
            
            next_obs = torch.tensor(np.vstack([
                agent_1_data['observations'][1:],
                agent_2_data['observations'][1:]
            ]), dtype=torch.float32)
            
            actions = torch.tensor(np.vstack([
                agent_1_data['actions'][:-1],
                agent_2_data['actions'][:-1]
            ]), dtype=torch.float32)
            
            all_obs.append(obs)
            all_next_obs.append(next_obs)
            all_actions.append(actions)
            
        return (torch.cat(all_obs, dim=0),
                torch.cat(all_next_obs, dim=0),
                torch.cat(all_actions, dim=0))

    def discriminator_loss(self, 
                          expert_obs: torch.Tensor,
                          expert_next_obs: torch.Tensor,
                          expert_actions: torch.Tensor,
                          policy_obs: torch.Tensor,
                          policy_next_obs: torch.Tensor,
                          policy_actions: torch.Tensor,
                          agent_idx: int) -> torch.Tensor:
        """Compute discriminator loss for a specific agent"""
        disc = self.discriminators[agent_idx]
        
        expert_scores = disc(expert_obs, expert_next_obs, expert_actions)
        policy_scores = disc(policy_obs, policy_next_obs, policy_actions)
        
        # Compute loss as in the paper
        expert_loss = -torch.mean(torch.log(torch.sigmoid(expert_scores)))
        policy_loss = -torch.mean(torch.log(1 - torch.sigmoid(policy_scores)))
        
        return expert_loss + policy_loss

    def policy_loss(self,
                   obs: torch.Tensor,
                   next_obs: torch.Tensor,
                   agent_idx: int) -> torch.Tensor:
        """Compute policy loss for a specific agent"""
        policy = self.policies[agent_idx]
        disc = self.discriminators[agent_idx]
        
        # Sample actions from policy
        dist = policy(obs)
        actions = dist.rsample()
        
        # Get reward signal from discriminator
        rewards = disc(obs, next_obs, actions)
        
        # Policy gradient loss
        policy_loss = -torch.mean(rewards)
        
        # Add entropy regularization
        entropy_loss = -torch.mean(dist.entropy())
        
        return policy_loss + 0.01 * entropy_loss

    def train_step(self, expert_trajectories: Dict, policy_trajectories: Dict):
        """Perform one training step"""
        # Process trajectory data
        expert_obs, expert_next_obs, expert_actions = self.process_trajectory_data(expert_trajectories)
        policy_obs, policy_next_obs, policy_actions = self.process_trajectory_data(policy_trajectories)
        
        # Update discriminators
        for i in range(self.n_agents):
            self.disc_optimizers[i].zero_grad()
            disc_loss = self.discriminator_loss(
                expert_obs, expert_next_obs, expert_actions,
                policy_obs, policy_next_obs, policy_actions,
                i
            )
            disc_loss.backward()
            self.disc_optimizers[i].step()
        
        # Update policies
        for i in range(self.n_agents):
            self.policy_optimizers[i].zero_grad()
            policy_loss = self.policy_loss(policy_obs, policy_next_obs, i)
            policy_loss.backward()
            self.policy_optimizers[i].step()
            
    def train(self, 
              expert_trajectories: Dict,
              n_epochs: int = 1000,
              batch_size: int = 64):
        """Main training loop"""
        for epoch in range(n_epochs):
            # Sample policy trajectories using current policies
            policy_trajectories = self.sample_trajectories(batch_size)
            
            # Perform training step
            self.train_step(expert_trajectories, policy_trajectories)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}")
                
    def sample_trajectories(self, batch_size: int) -> Dict:
        """
        Sample trajectories using current policies
        This should be implemented based on your specific environment
        """
        # Placeholder - implement based on your environment
        env = gym.make("Foraging-8x8-2p-3f-v3")
        


        pass

    def save(self, path: str):
        """Save the model"""
        torch.save({
            'discriminators': self.discriminators.state_dict(),
            'policies': self.policies.state_dict(),
        }, path)

    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path)
        self.discriminators.load_state_dict(checkpoint['discriminators'])
        self.policies.load_state_dict(checkpoint['policies'])

if __name__ == "__main__":
    # Load expert trajectories
    with open('../../../trajectories/trajectories_20250114_085245.pkl', 'rb') as f:
        expert_trajectories = pickle.load(f)

    print("expert data size: ",len(expert_trajectories['trajectories']))
        
    # Initialize MAAIRL agent
    agent = MAAIRL(obs_dim=15, act_dim=6)
    print(agent)
    print(agent.policies)
    print(agent.discriminators)
    # Train the agent
    # agent.train(expert_trajectories)
    
    # Save the model
    agent.save('maairl_model.pth')

    