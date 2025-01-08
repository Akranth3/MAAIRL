import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# Neural network architectures
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)

class DiscriminatorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Reward estimator g_w
        self.reward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # State-only potential function h_φ
        self.potential = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action, next_state, gamma=0.99):
        sa = torch.cat([state, action], dim=-1)
        reward = self.reward(sa)
        potential_curr = self.potential(state)
        potential_next = self.potential(next_state)
        
        # f_w = g_w(s,a) + γh_φ(s') - h_φ(s)
        return reward + gamma * potential_next - potential_curr

class MAAIRL:
    def __init__(
        self,
        num_agents,
        state_dim,
        action_dims,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.device = device
        
        # Initialize networks for each agent
        self.generators = []
        self.discriminators = []
        self.generator_optimizers = []
        self.discriminator_optimizers = []
        
        for i in range(num_agents):
            # Generator (policy) networks
            generator = PolicyNetwork(state_dim, action_dims[i]).to(device)
            self.generators.append(generator)
            self.generator_optimizers.append(
                torch.optim.Adam(generator.parameters(), lr=learning_rate)
            )
            
            # Discriminator networks
            discriminator = DiscriminatorNetwork(state_dim, action_dims[i]).to(device)
            self.discriminators.append(discriminator)
            self.discriminator_optimizers.append(
                torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
            )

    def update_discriminators(self, expert_batch, policy_batch):
        """Update discriminators using expert and policy trajectories"""
        discriminator_losses = []
        
        for i in range(self.num_agents):
            # Get current agent's data
            expert_states = expert_batch['states']
            expert_actions = expert_batch['actions'][:, i]
            expert_next_states = expert_batch['next_states']
            
            policy_states = policy_batch['states']
            policy_actions = policy_batch['actions'][:, i]
            policy_next_states = policy_batch['next_states']
            
            # Move to device
            expert_states = torch.FloatTensor(expert_states).to(self.device)
            expert_actions = torch.FloatTensor(expert_actions).to(self.device)
            expert_next_states = torch.FloatTensor(expert_next_states).to(self.device)
            
            policy_states = torch.FloatTensor(policy_states).to(self.device)
            policy_actions = torch.FloatTensor(policy_actions).to(self.device)
            policy_next_states = torch.FloatTensor(policy_next_states).to(self.device)
            
            # Calculate discriminator outputs
            expert_scores = self.discriminators[i](expert_states, expert_actions, expert_next_states)
            policy_scores = self.discriminators[i](policy_states, policy_actions, policy_next_states)
            
            # Calculate discriminator loss (Equation 11 in the paper)
            expert_loss = -torch.mean(
                torch.log(torch.sigmoid(expert_scores) + 1e-8)
            )
            policy_loss = -torch.mean(
                torch.log(1 - torch.sigmoid(policy_scores) + 1e-8)
            )
            discriminator_loss = expert_loss + policy_loss
            
            # Update discriminator
            self.discriminator_optimizers[i].zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizers[i].step()
            
            discriminator_losses.append(discriminator_loss.item())
            
        return np.mean(discriminator_losses)

    def update_generators(self, states, actions, next_states):
        """Update generator policies using discriminator feedback"""
        generator_losses = []
        
        for i in range(self.num_agents):
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions[:, i]).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            
            # Calculate discriminator scores
            scores = self.discriminators[i](states_tensor, actions_tensor, next_states_tensor)
            
            # Calculate generator loss (Equation 12 in the paper)
            # Maximize log D(s,a) - log(1-D(s,a)) which is equivalent to maximizing f_w(s,a) - log(q_θ)
            log_probs = torch.log(self.generators[i](states_tensor) + 1e-8)
            generator_loss = -torch.mean(scores - log_probs)
            
            # Update generator
            self.generator_optimizers[i].zero_grad()
            generator_loss.backward()
            self.generator_optimizers[i].step()
            
            generator_losses.append(generator_loss.item())
            
        return np.mean(generator_losses)

    def get_actions(self, states, deterministic=False):
        """Get actions from all policies"""
        actions = []
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        for i in range(self.num_agents):
            action_probs = self.generators[i](states_tensor)
            
            if deterministic:
                actions.append(torch.argmax(action_probs, dim=-1))
            else:
                dist = torch.distributions.Categorical(action_probs)
                actions.append(dist.sample())
                
        return torch.stack(actions, dim=1).cpu().numpy()

    def get_reward_functions(self):
        """Return the learned reward functions"""
        return [discriminator.reward for discriminator in self.discriminators]

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, next_states = map(np.stack, zip(*samples))
        return {
            'states': states,
            'actions': actions,
            'next_states': next_states
        }
    
    def __len__(self):
        return len(self.buffer)