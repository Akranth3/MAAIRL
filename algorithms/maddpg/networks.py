import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, fc1_dims=64, fc2_dims=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action = nn.Linear(fc2_dims, action_dims)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.sigmoid(self.action(x))
        return action

class CriticNetwork(nn.Module):
    def __init__(self, state_dims, action_dims, fc1_dims=64, fc2_dims=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(sum(state_dims) + action_dims, fc1_dims) #removed sum from action space as it is only an int.
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        
    def forward(self, states, actions):
        state_cat = torch.cat(states, dim=1)
        action_cat = torch.cat(actions, dim=1)
        state_action = torch.cat([state_cat, action_cat], dim=1)
        
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q