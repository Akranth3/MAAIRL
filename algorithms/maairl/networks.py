import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, state, action):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        return log_prob

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RewardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.network(state_action).squeeze(-1)

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.reward = RewardNetwork(state_dim, action_dim, hidden_dim)
        self.value = RewardNetwork(state_dim, 0, hidden_dim)  # State-only value network

    def forward(self, state, action, next_state):
        reward = self.reward(state, action)
        value = self.value(state, torch.tensor([]))
        next_value = self.value(next_state, torch.tensor([]))
        return reward + 0.99 * next_value - value  # γ=0.99