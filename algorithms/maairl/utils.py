import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
    
    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        
        # Convert to tensors and handle multi-agent structure
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        states = [torch.FloatTensor(np.stack([s[i] for s in state])).to(device) 
                 for i in range(len(state[0]))]
        actions = [torch.FloatTensor(np.stack([a[i] for a in action])).to(device) 
                  for i in range(len(action[0]))]
        next_states = [torch.FloatTensor(np.stack([ns[i] for ns in next_state])).to(device) 
                      for i in range(len(next_state[0]))]
        rewards = [torch.FloatTensor(np.stack([r[i] for r in reward])).to(device) 
                  for i in range(len(reward[0]))]
        dones = torch.FloatTensor(done).to(device)
        
        return {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'rewards': rewards,
            'dones': dones
        }
    
    def __len__(self):
        return len(self.buffer)

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(rewards)
    lastgae = 0
    
    for t in reversed(range(len(rewards))):
        lastgae = deltas[t] + gamma * lam * (1 - dones[t]) * lastgae
        advantages[t] = lastgae
    
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns