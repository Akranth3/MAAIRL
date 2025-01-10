# config.py
import torch
import argparse
from pettingzoo.mpe import simple_adversary_v3
from matplotlib import pyplot as plt
import pickle
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Independent Q-Learning Training")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="Exploration rate decay")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum exploration rate")
    return parser.parse_args()

args = get_args()

class Config:
    def __init__(self):
        # Environment settings
        self.n_agents = 3
        self.state_dim = [8, 10, 10]  # State dimensions for each agent
        self.action_dim = 5
        
        # Training hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.batch_size = 64
        self.target_update = 200  # Update target network every N steps
        self.memory_size = 100000
        self.hidden_dim = 64
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# networks.py
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

# memory.py
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.memory)

# agent.py
class IQLAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-Networks (online and target)
        self.q_network = QNetwork(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.target_network = QNetwork(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.memory_size)
        self.steps = 0
        
    def select_action(self, state, epsilon):
        '''
        the function directly returns the best action by finding the corresponding max value in the output of Q-network.
        '''
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)

            q_values = self.q_network(state)
            # print("the return of select_action, ",q_values.max(1)[1].item(), q_values.shape)
            return q_values.max(1)[1].item()
    
    def update(self):
        if len(self.memory) < self.config.batch_size:
            # print("memomry size is less thatn batch size, so not updating.")
            return
        
        # Sample from memory
        batch = self.memory.sample(self.config.batch_size)
        
        # Convert to tensor
       
        states = torch.FloatTensor(np.array(batch.state)).to(self.config.device)
        actions = torch.LongTensor(batch.action).to(self.config.device)
        rewards = torch.FloatTensor(batch.reward).to(self.config.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.config.device)
        dones = torch.FloatTensor(batch.done).to(self.config.device)

        # print("hello")
        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(next_states.shape)
        # print(dones.shape)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        # print("q-values: ", current_q_values.shape)

        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * max_next_q_values
        
        # print("max next q shape, ", max_next_q_values.shape)
        # print("target q-shape, ", target_q_values.shape)
        
        # Compute loss and update
        # loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

# main.py
def train():
    config = Config()
    env = simple_adversary_v3.env(max_cycles=100)
    
    # Initialize agents
    agents = [IQLAgent(config.state_dim[i], config.action_dim, config) 
              for i in range(config.n_agents)]
    
    agent_to_idx = {"adversary_0": 0, "agent_0": 1, "agent_1": 2}
    epsilon = args.epsilon
    plot_rewards = []
    log_rewards = []

    
    for episode in range(args.episodes):
        env.reset()
        episode_reward = 0
        agent_wise_rewards = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
        losss = [0,0,0]
        observations = []
        next_observation = []
        actions = []
        rewards = []
        dones = []
        count = 0
        # Main training loop
        for agent in env.agent_iter():
            observation, reward, done, truncated, info = env.last()
            
            # Select and perform action
            action = agents[agent_to_idx[agent]].select_action(observation, epsilon)

            if done or truncated:
                break
                
            # Step environment
            env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            count+=1

            if count == 3:
                next_observation = [env.observe(temp_agent) for temp_agent in env.agents]
                # Store transition in memory
                i = 0
                print("observations: ", observations)
                print("next observations: ", next_observation)
                for temp_agent in env.agents:
                    agents[agent_to_idx[temp_agent]].memory.push(
                        observations[i],
                        actions[i],
                        rewards[i],
                        next_observation[i],
                        dones[i]
                    )
                    i+=1
                observations = []
                actions = []
                rewards = []
                dones = []
                count = 0
                
            # Update the network
            loss = agents[agent_to_idx[agent]].update()

            losss[agent_to_idx[agent]] = loss
            episode_reward += reward
            agent_wise_rewards[agent] += reward
            
        
        # Decay epsilon
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        
        # Log rewards
        plot_rewards.append(episode_reward)
        log_rewards.append(agent_wise_rewards)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}, Epsilon: {epsilon:.3f}, epsidoe length: {env.steps}, loss: {losss}")
    
    # Save results
    plt.figure(figsize=(15, 5))
    plt.plot(plot_rewards)
    plt.savefig("rewards.png")
    plt.close()
    pickle.dump(plot_rewards, open("plot_rewards.pkl", "wb"))

    
    return agents

if __name__ == "__main__":
    agents = train()
    # Save trained models
    for i in range(3):
        torch.save(agents[i].q_network.state_dict(), f"check_points/q_network_agent_{i}.pth")