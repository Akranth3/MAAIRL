import gymnasium as gym
from gymnasium.envs.registration import register
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
from torch.optim import Adam
import os

def get_args():
    parser = argparse.ArgumentParser(description="IPPO for LBForaging environments")
    parser.add_argument("--episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--epsilon_clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--value_coef", type=float, default=0.5, help="value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--update_epochs", type=int, default=4, help="number of PPO epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--save_dir", type=str, default="networks", help="directory to save networks")
    return parser.parse_args()

class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=64):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.actor(x), self.critic(x)
    
    def get_action(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def get_batch(self, batch_size=None):
        batch_size = batch_size or len(self.states)
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        return (
            torch.FloatTensor(np.array(self.states)[indices]),
            torch.LongTensor(np.array(self.actions)[indices]),
            torch.FloatTensor(np.array(self.rewards)[indices]),
            torch.FloatTensor(np.array(self.log_probs)[indices]),
            torch.FloatTensor(np.array(self.values)[indices]),
            torch.FloatTensor(np.array(self.dones)[indices])
        )

def register_env(size=8, players=2, food=1, force_coop=False):
    register(
        id=f"Foraging-{size}x{size}-{players}p-{food}f{'-coop' if force_coop else ''}-v3",
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": players,
            "min_player_level": 1,
            "max_player_level": 3,
            "min_food_level": 1,
            "max_food_level": 3,
            "field_size": (size, size),
            "max_num_food": food,
            "sight": size,
            "max_episode_steps": 50,
            "force_coop": force_coop,
            "normalize_reward": True,
            "grid_observation": False,
            "observe_agent_levels": True,
            "penalty": 0.0,
            "render_mode": None
        }
    )

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = []
    gae = 0
    
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value = values[step + 1]
            
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
        
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns

def train_agent(memory, actor_critic, optimizer, device, args):
    states, actions, rewards, old_log_probs, values, dones = memory.get_batch()
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    
    # Compute GAE
    with torch.no_grad():
        _, next_value = actor_critic(torch.FloatTensor(memory.states[-1]).unsqueeze(0).to(device))
        next_value = next_value.item()
    
    advantages, returns = compute_gae(
        rewards, values, dones, next_value,
        args.gamma, args.gae_lambda
    )
    advantages = advantages.to(device)
    returns = returns.to(device)
    
    total_loss = 0
    for _ in range(args.update_epochs):
        # Get new log probs, values, and entropy
        log_probs, values, entropy = actor_critic.evaluate_actions(states, actions)
        values = values.to(device)
        
        # Compute ratio and surrogate loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - args.epsilon_clip, 1 + args.epsilon_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        
        # Compute entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = actor_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / args.update_epochs

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Environment setup
    register_env()
    env = gym.make("Foraging-8x8-2p-1f-v3")
    
    # Initialize networks and optimizers for each agent
    num_agents = 2
    obs_dim = 9  # Observation space size for each agent
    action_dim = 6  # Action space size for each agent
    
    actor_critics = [ActorCritic(obs_dim, action_dim).to(device) for _ in range(num_agents)]
    optimizers = [Adam(ac.parameters(), lr=args.lr) for ac in actor_critics]
    memories = [PPOMemory() for _ in range(num_agents)]
    
    success_episodes = 0
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_rewards = np.zeros(num_agents)
        step_count = 0
        
        # Episode loop
        while True:
            actions = []
            log_probs = []
            values = []
            
            # Get actions for each agent
            for i in range(num_agents):
                action, log_prob, value = actor_critics[i].get_action(obs[i], device)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            # Environment step
            next_obs, rewards, done, truncated, _ = env.step(actions)
            episode_rewards += np.array(rewards)
            step_count += 1
            
            # Store transitions for each agent
            for i in range(num_agents):
                memories[i].push(
                    obs[i], actions[i], rewards[i], log_probs[i], values[i], done
                )
            
            obs = next_obs
            
            if done or truncated:
                break
        
        # Train agents
        losses = []
        for i in range(num_agents):
            loss = train_agent(memories[i], actor_critics[i], optimizers[i], device, args)
            losses.append(loss)
            memories[i].clear()
        
        if step_count < 50:
            success_episodes += 1
        
        # Logging
        if episode % 100 == 0:
            avg_loss = np.mean(losses)
            print(f"Episode: {episode + 1}, Rewards: {episode_rewards}, Steps: {step_count}, "
                  f"Avg Loss: {avg_loss:.4f}")
        
    # Save final networks
    for i, actor_critic in enumerate(actor_critics):
        torch.save(actor_critic.state_dict(), 
                  os.path.join(args.save_dir, f"actor_critic_{i}_final.pth"))
    
    print(f"Success rate: {success_episodes/args.episodes*100:.2f}%")
    print("Training completed. Networks saved.")

if __name__ == "__main__":
    main()