import gymnasium as gym
from gymnasium.envs.registration import register
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random
from torch.optim import Adam
import os

def get_args():
    parser = argparse.ArgumentParser(description="IQL for LBForaging environments")
    parser.add_argument("--episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="starting epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="minimum epsilon for exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="epsilon decay rate")
    parser.add_argument("--target_update", type=int, default=100, help="target network update frequency")
    parser.add_argument("--buffer_size", type=int, default=20000, help="replay buffer size")
    parser.add_argument("--save_dir", type=str, default="networks", help="directory to save networks")
    return parser.parse_args()

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return None
        return Transition(*zip(*random.sample(self.memory, batch_size)))
    
    def __len__(self):
        return len(self.memory)

class MultiAgentFCNetwork(nn.Module):
    def __init__(self, in_sizes: list[int], out_sizes: list[int], hidden_dims=(64, 64)):
        super().__init__()
        self.networks = nn.ModuleList()
        
        for in_size, out_size in zip(in_sizes, out_sizes):
            network = nn.Sequential(
                nn.Linear(in_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], out_size)
            )
            self.networks.append(network)

    def forward(self, inputs: list[torch.Tensor]):
        return [network(x) for x, network in zip(inputs, self.networks)]

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

def select_action(state, q_network, epsilon, device, action_space=6):
    if random.random() > epsilon:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            # print(q_values)
            return q_values.argmax(1).item()
    else:
        return random.randint(0, action_space - 1)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Environment setup
    register_env()
    env = gym.make("Foraging-8x8-2p-1f-v3")
    
    # Network initialization
    obs_sizes = [9, 9]  # Observation space size for each agent
    action_sizes = [6, 6]  # Action space size for each agent
    
    q_networks = MultiAgentFCNetwork(obs_sizes, action_sizes).to(device)
    target_q_networks = MultiAgentFCNetwork(obs_sizes, action_sizes).to(device)
    target_q_networks.load_state_dict(q_networks.state_dict())
    
    optimizers = [Adam(network.parameters(), lr=args.lr) for network in q_networks.networks]
    agent_wise_replay_buffer = [ReplayBuffer(args.buffer_size) for _ in range(len(obs_sizes))]
    
    epsilon = args.epsilon_start
    total_steps = 0
    success_episodes = 0
    suceess_after_epsilon_dim = 0

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_rewards = np.zeros(len(obs_sizes))
        episode_loss = 0
        step_count = 0

        agent_1_actions = []
        agent_2_actions = []        
        
        while True:
            actions = []
            for i, agent_obs in enumerate(obs):
                action = select_action(agent_obs, q_networks.networks[i], epsilon, device)
                actions.append(action)
                if i == 0:
                    agent_1_actions.append(action)
                else:
                    agent_2_actions.append(action)
            # actions = 
            
            
            next_obs, rewards, done, truncated, _ = env.step(actions)
            episode_rewards += np.array(rewards)
            step_count += 1
            
            # Store transitions
            for i in range(len(actions)):
                agent_wise_replay_buffer[i].push(
                    obs[i], actions[i], rewards[i], next_obs[i], done
                )
            
            # Training
            for i in range(len(actions)):
                batch = agent_wise_replay_buffer[i].sample(args.batch_size)
                if batch is not None:
                    states = torch.FloatTensor(np.array(batch.state)).to(device)
                    actions = torch.LongTensor(batch.action).view(-1, 1).to(device)
                    rewards = torch.FloatTensor(batch.reward).view(-1, 1).to(device)
                    next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
                    dones = torch.FloatTensor(batch.done).view(-1, 1).to(device)
                    
                    # Zero gradients
                    optimizers[i].zero_grad()
                    
                    # Current Q values
                    current_q_values = q_networks.networks[i](states).gather(1, actions)
                    
                    # Target Q values
                    with torch.no_grad():
                        next_q_values = target_q_networks.networks[i](next_states).max(1)[0].unsqueeze(1)
                        target_q_values = rewards + args.gamma * next_q_values * (1 - dones)
                    
                    # Compute loss and update
                    loss = nn.MSELoss()(current_q_values, target_q_values)
                    loss.backward()
                    optimizers[i].step()
                    
                    episode_loss += loss.item()
            
            # Update target network
            if total_steps % args.target_update == 0:
                target_q_networks.load_state_dict(q_networks.state_dict())
                # print("@@@@@@@@@@@22222")
                # print(target_q_networks.state_dict())
                # print("##############")
                # print(q_networks.state_dict())
            # for target_param, local_param in zip(target_q_network.parameters(), q_network.parameters()):
            #     target_param.data.copy_(τ * local_param.data + (1.0 - τ) * target_param.data)

            
            total_steps += 1
            obs = next_obs
            
            if done or truncated:
                break
        
        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        
        variance_action_1 = np.var(agent_1_actions)
        variance_action_2 = np.var(agent_2_actions)

        if step_count < 50:
            success_episodes += 1
            if epsilon < 0.2:
                suceess_after_epsilon_dim+=1

        # Logging
        if episode % 100 == 0:
            avg_loss = episode_loss / step_count if step_count > 0 else 0
            print(f"Episode: {episode + 1}, Rewards: {episode_rewards}, Steps: {step_count}, "
                f"Epsilon: {epsilon:.3f}, Avg Loss: {avg_loss:.4f}, variance {variance_action_1:.2f}, {variance_action_2:.2f}")
        
        # Save networks periodically
        # if (episode + 1) % 100 == 0:
        #     for i, network in enumerate(q_networks.networks):
        #         torch.save(network.state_dict(), 
        #                  os.path.join(args.save_dir, f"q_network_{i}_episode_{episode+1}.pth"))
    
    # Save final networks
    for i, network in enumerate(q_networks.networks):
        torch.save(network.state_dict(), 
                  os.path.join(args.save_dir, f"q_network_{i}_final_30000.pth"))
    print("success_episodes percentage: ", success_episodes/args.episodes*100)
    print("success_episodes percentage after epsilon dim: ", suceess_after_epsilon_dim/args.episodes*100)
    print("Training completed. Networks saved.")




if __name__ == "__main__":
    main()