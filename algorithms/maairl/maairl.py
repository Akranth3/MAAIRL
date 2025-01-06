import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class Discriminator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Reward approximator g_w
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # State-only potential function h_φ
        self.potential_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.reward_net(sa)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                next_state: torch.Tensor, gamma: float) -> torch.Tensor:
        # f_w,φ(s,a,s') = g_w(s,a) + γh_φ(s') - h_φ(s)
        return (self.reward(state, action) + 
                gamma * self.potential_net(next_state) - 
                self.potential_net(state))

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std.exp()
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

class MAAIRL:
    def __init__(
        self,
        state_dim: int,
        action_dims: Dict[str, int],
        agent_ids: List[str],
        lr_policy: float = 3e-4,
        lr_disc: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 256
    ):
        self.agent_ids = agent_ids
        self.gamma = gamma
        
        # Initialize networks for each agent
        self.policies = {
            agent_id: PolicyNetwork(state_dim, action_dims[agent_id], hidden_dim)
            for agent_id in agent_ids
        }
        
        self.discriminators = {
            agent_id: Discriminator(state_dim, action_dims[agent_id], hidden_dim)
            for agent_id in agent_ids
        }
        
        # Initialize optimizers
        self.policy_optimizers = {
            agent_id: optim.Adam(self.policies[agent_id].parameters(), lr=lr_policy)
            for agent_id in agent_ids
        }
        
        self.disc_optimizers = {
            agent_id: optim.Adam(self.discriminators[agent_id].parameters(), lr=lr_disc)
            for agent_id in agent_ids
        }
    
    def update_discriminator(
        self,
        agent_id: str,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        expert_next_states: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        policy_next_states: torch.Tensor
    ) -> float:
        # Get discriminator outputs
        expert_scores = self.discriminators[agent_id](
            expert_states, expert_actions, expert_next_states, self.gamma
        )
        policy_scores = self.discriminators[agent_id](
            policy_states, policy_actions, policy_next_states, self.gamma
        )
        
        # Compute discriminator loss
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_scores, torch.ones_like(expert_scores)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_scores, torch.zeros_like(policy_scores)
        )
        disc_loss = expert_loss + policy_loss
        
        # Update discriminator
        self.disc_optimizers[agent_id].zero_grad()
        disc_loss.backward()
        self.disc_optimizers[agent_id].step()
        
        return disc_loss.item()
    
    def update_policy(
        self,
        agent_id: str,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> float:
        # Sample actions from current policy
        new_actions, log_probs = self.policies[agent_id].sample(states)
        
        # Get reward from discriminator
        rewards = self.discriminators[agent_id].reward(states, new_actions)
        
        # Compute policy loss (negative of the reward plus entropy regularization)
        policy_loss = -(rewards + 0.01 * log_probs).mean()
        
        # Update policy
        self.policy_optimizers[agent_id].zero_grad()
        policy_loss.backward()
        self.policy_optimizers[agent_id].step()
        
        return policy_loss.item()

def process_demonstrations(demonstrations: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert demonstrations to tensors for training."""
    processed = defaultdict(list)
    
    for demo in demonstrations:
        for key, value in demo.items():
            processed[key].append(value)
    
    return {k: torch.FloatTensor(np.array(v)) for k, v in processed.items()}

def train_maairl(
    expert_demonstrations: List[Dict],
    state_dim: int,
    action_dims: Dict[str, int],
    agent_ids: List[str],
    n_epochs: int = 1000,
    batch_size: int = 64
) -> Tuple[Dict[str, PolicyNetwork], Dict[str, Discriminator]]:
    """
    Train MA-AIRL using expert demonstrations.
    
    Args:
        expert_demonstrations: List of dictionaries containing expert trajectories
        state_dim: Dimension of the state space
        action_dims: Dictionary mapping agent IDs to their action space dimensions
        agent_ids: List of agent IDs
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Tuple of trained policies and discriminators for each agent
    """
    
    # Initialize MA-AIRL
    maairl = MAAIRL(state_dim, action_dims, agent_ids)
    
    # Process demonstrations
    demos = process_demonstrations(expert_demonstrations)
    
    # Training loop
    for epoch in range(n_epochs):
        # Sample random batch
        idx = np.random.randint(0, len(expert_demonstrations), batch_size)
        
        expert_batch = {
            k: v[idx] for k, v in demos.items()
        }
        
        # Generate policy trajectories
        policy_batch = defaultdict(list)
        for states in expert_batch['states']:
            actions = {}
            for agent_id in agent_ids:
                action, _ = maairl.policies[agent_id].sample(states)
                actions[agent_id] = action
            policy_batch['actions'].append(actions)
            
        policy_batch = {k: torch.stack(v) for k, v in policy_batch.items()}
        
        # Update discriminators and policies for each agent
        for agent_id in agent_ids:
            # Update discriminator
            disc_loss = maairl.update_discriminator(
                agent_id,
                expert_batch['states'],
                expert_batch['actions'][agent_id],
                expert_batch['next_states'],
                expert_batch['states'],
                policy_batch['actions'][agent_id],
                expert_batch['next_states']
            )
            
            # Update policy
            policy_loss = maairl.update_policy(
                agent_id,
                expert_batch['states'],
                policy_batch['actions'][agent_id],
                expert_batch['next_states']
            )
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Agent {agent_id}:")
                print(f"Discriminator loss: {disc_loss:.4f}")
                print(f"Policy loss: {policy_loss:.4f}")
    
    return maairl.policies, maairl.discriminators

# Example usage:
def get_policy_and_rewards(
    expert_demonstrations: List[Dict],
    state_dim: int,
    action_dims: Dict[str, int],
    agent_ids: List[str]
) -> Tuple[Dict[str, PolicyNetwork], Dict[str, Discriminator]]:
    """
    Main function to train MA-AIRL and get policies and reward functions.
    
    Args:
        expert_demonstrations: List of dictionaries containing expert trajectories
        state_dim: Dimension of the state space
        action_dims: Dictionary mapping agent IDs to their action space dimensions
        agent_ids: List of agent IDs
        
    Returns:
        Tuple of (policies, reward_functions) for each agent
    """
    # Train MA-AIRL
    policies, discriminators = train_maairl(
        expert_demonstrations,
        state_dim,
        action_dims,
        agent_ids
    )
    
    return policies, discriminators