import torch
import torch.nn.functional as F
import numpy as np
from .networks import PolicyNetwork, Discriminator

class MAAIRL:
    def __init__(self, num_agents, state_dims, action_dims, device='cuda'):
        self.num_agents = num_agents
        self.device = device
        
        # Initialize networks for each agent
        self.policies = [PolicyNetwork(state_dims[i], action_dims[i]).to(device) 
                        for i in range(num_agents)]
        self.discriminators = [Discriminator(state_dims[i], action_dims[i]).to(device) 
                             for i in range(num_agents)]
        
        # Initialize optimizers
        self.policy_optimizers = [torch.optim.Adam(policy.parameters(), lr=3e-4) 
                                for policy in self.policies]
        self.discriminator_optimizers = [torch.optim.Adam(disc.parameters(), lr=3e-4) 
                                       for disc in self.discriminators]

    def update_discriminator(self, expert_states, expert_actions, expert_next_states,
                           policy_states, policy_actions, policy_next_states):
        disc_losses = []
        
        for i in range(self.num_agents):
            # Expert data
            expert_logits = self.discriminators[i](
                expert_states[i], expert_actions[i], expert_next_states[i])
            expert_loss = F.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits))
            
            # Policy data
            policy_logits = self.discriminators[i](
                policy_states[i], policy_actions[i], policy_next_states[i])
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits))
            
            # Total discriminator loss
            disc_loss = expert_loss + policy_loss
            
            # Update discriminator
            self.discriminator_optimizers[i].zero_grad()
            disc_loss.backward()
            self.discriminator_optimizers[i].step()
            
            disc_losses.append(disc_loss.item())
            
        return np.mean(disc_losses)

    def update_policy(self, states, actions, next_states):
        policy_losses = []
        
        for i in range(self.num_agents):
            # Get current log probabilities
            log_probs = self.policies[i].get_log_prob(states[i], actions[i])
            
            # Get rewards from discriminator
            rewards = self.discriminators[i](states[i], actions[i], next_states[i])
            
            # Policy loss (negative because we want to maximize reward)
            policy_loss = -(log_probs * rewards).mean()
            
            # Update policy
            self.policy_optimizers[i].zero_grad()
            policy_loss.backward()
            self.policy_optimizers[i].step()
            
            policy_losses.append(policy_loss.item())
            
        return np.mean(policy_losses)

    def get_actions(self, states):
        actions = []
        log_probs = []
        
        for i in range(self.num_agents):
            action, log_prob = self.policies[i].sample_action(states[i])
            actions.append(action)
            log_probs.append(log_prob)
            
        return actions, log_probs

    def save(self, path):
        save_dict = {
            'policies': [policy.state_dict() for policy in self.policies],
            'discriminators': [disc.state_dict() for disc in self.discriminators]
        }
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path)
        for i, policy_state_dict in enumerate(save_dict['policies']):
            self.policies[i].load_state_dict(policy_state_dict)
        for i, disc_state_dict in enumerate(save_dict['discriminators']):
            self.discriminators[i].load_state_dict(disc_state_dict)