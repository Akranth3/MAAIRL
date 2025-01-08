# config.py
import torch
from pettingzoo.mpe import simple_adversary_v3

class Config:
    def __init__(self):
        # Environment settings
        self.n_agents = 3
        self.state_dim = [8, 10, 10]
        self.action_dim = 5
        
        # Training hyperparameters
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.update_timestep = 2000
        self.hidden_dim = 64
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.actor(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.critic(state)

# memory.py
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done', 'action_prob'))

class Memory:
    def __init__(self):
        self.memories = []
        
    def push(self, *args):
        self.memories.append(Transition(*args))
        
    def clear(self):
        self.memories = []
        
    def get_batch(self):
        batch = Transition(*zip(*self.memories))
        return batch

# agent.py
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class MAPPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.critic = Critic(state_dim, config.hidden_dim).to(config.device)
        self.memory = Memory()
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.config.device)
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_prob = action_probs[action].item()
            
        return action.item(), action_prob
    
    def update(self):
        batch = self.memory.get_batch()
        
        # Convert to tensor
        states = torch.FloatTensor(batch.state).to(self.config.device)
        actions = torch.LongTensor(batch.action).to(self.config.device)
        rewards = torch.FloatTensor(batch.reward).to(self.config.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.config.device)
        old_probs = torch.FloatTensor(batch.action_prob).to(self.config.device)
        
        # Calculate advantages
        with torch.no_grad():
            next_values = self.critic(next_states)
            current_values = self.critic(states)
            advantages = rewards + self.config.gamma * next_values * (1 - torch.FloatTensor(batch.done).to(self.config.device)) - current_values
            
        # PPO update for k epochs
        for _ in range(self.config.k_epochs):
            # Actor loss
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_probs = dist.log_prob(actions).exp()
            
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.config.eps_clip, 1+self.config.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(self.critic(states), rewards + self.config.gamma * next_values * (1 - torch.FloatTensor(batch.done).to(self.config.device)))
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        self.memory.clear()

# main.py
# import gym
import numpy as np

def train():
    config = Config()
    env = simple_adversary_v3.env()  # Make sure you have the environment installed

    agents = [MAPPOAgent(config.state_dim[i], config.action_dim, config) 
              for i in range(config.n_agents)]
    
    print(agents)
    agent_to_idx = {"adversary_0":0, "agent_0":1, "agent_1":2}
    log_rewards = []
    time_step = 0
    for episode in range(25000):  # Number of episodes
        env.reset() # does not retuen anything.
        episode_reward = 0
        done = False
        
        # while not done:
        actions = []
        action_probs = []
        states = []
        next_states = []
        rewards = []

        agent_wise_rewards = {"adversary_0":0, "agent_0":0, "agent_1":0}
        
        # Get actions for all agents
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
                        done,
                        action_probs[i]
                    )
                states = []
                next_states = []
                rewards = []
                actions = []
                action_probs = []
            
            episode_reward += reward
            agent_wise_rewards[agent] += reward
            time_step += 1
        
            # Update if enough steps have been accumulated
            if time_step % config.update_timestep == 0:
                for agent in agents:
                    agent.update()
                    print("agent updated, episode: ", episode)
        log_rewards.append(agent_wise_rewards)
        # print(f"Episode {episode}, Total Reward: {episode_reward}")
    
    return agents

if __name__ == "__main__":
    agents = train()
    for i in range(3):
        torch.save(agents[i].actor.state_dict(), f"check_points/actor_agent_{i}.pth")
        torch.save(agents[i].critic.state_dict(), f"check_points/critic_agent_{i}.pth")