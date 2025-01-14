import torch
import numpy as np
import pickle
import gymnasium as gym
import lbforaging
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
HIDDEN_DIM = 256
NUM_EPOCHS = 10

class RewardNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, obs, actions):
        input_tensor = torch.cat([obs, actions], dim=-1)
        return self.network(input_tensor)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.reward_net = RewardNetwork(obs_dim, act_dim)
        self.value_net = RewardNetwork(obs_dim, 0)  # State-only network

    def forward(self, obs, next_obs, actions):
        reward = self.reward_net(obs, actions)
        next_value = self.value_net(next_obs, None)
        current_value = self.value_net(obs, None)
        return reward + GAMMA * next_value - current_value

def load_expert_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def sample_trajectory(env, policy):
    obs, _ = env.reset()
    done = False
    trajectory = []
    while not done:
        state = torch.tensor(np.concatenate(obs), dtype=torch.float32)
        action_probs = policy(state)
        actions = [torch.multinomial(action_probs[:env.action_space.n], 1).item()
                   for _ in range(env.n_agents)]
        next_obs, reward, done, _, _ = env.step(actions)
        trajectory.append((obs, actions, reward, next_obs))
        obs = next_obs
    return trajectory

def compute_loss(discriminator, policy, expert_data, env):
    policy_trajectories = [sample_trajectory(env, policy) for _ in range(BATCH_SIZE)]
    
    expert_obs = torch.cat([torch.tensor(d[0], dtype=torch.float32) for traj in expert_data for d in traj])
    expert_actions = torch.cat([torch.tensor(d[1], dtype=torch.float32) for traj in expert_data for d in traj])
    
    policy_obs = torch.cat([torch.tensor(d[0], dtype=torch.float32) for traj in policy_trajectories for d in traj])
    policy_actions = torch.cat([torch.tensor(d[1], dtype=torch.float32) for traj in policy_trajectories for d in traj])

    expert_scores = discriminator(expert_obs, None, expert_actions)
    policy_scores = discriminator(policy_obs, None, policy_actions)

    expert_loss = torch.mean(-torch.log(expert_scores + 1e-8))
    policy_loss = torch.mean(-torch.log(1 - policy_scores + 1e-8))

    return expert_loss + policy_loss

def train(env_name, expert_data_path):
    env = gym.make(env_name)
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(2 * obs_dim, act_dim)
    discriminator = Discriminator(obs_dim, act_dim)

    policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    expert_data = load_expert_data(expert_data_path)

    for epoch in range(NUM_EPOCHS):
        discriminator_optimizer.zero_grad()
        loss = compute_loss(discriminator, policy, expert_data, env)
        loss.backward()
        discriminator_optimizer.step()

        policy_optimizer.zero_grad()
        policy_loss = -torch.mean(discriminator(torch.tensor(env.reset()[0]), None, policy))
        policy_loss.backward()
        policy_optimizer.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    ENV_NAME = "Foraging-8x8-2p-3f-v3"
    EXPERT_DATA_PATH = "../../../trajectories/trajectories_20250114_110817.pkl"
    train(ENV_NAME, EXPERT_DATA_PATH)
