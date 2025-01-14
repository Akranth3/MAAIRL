import torch
import numpy as np
import pickle
import gymnasium as gym
import lbforaging
import torch.nn as nn
import numpy as np
import scipy
import pandas as pd


class RewardNetwork(nn.Module):
    '''
    input will be state and action and the output will be the reward. 
    each agent has its own reward network.
    '''
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # since the reward is between 0 and 1 [[need to check]]
        )
        
    def forward(self, obs, actions):
        input_tensor = torch.cat([obs, actions], dim=-1)
        return self.network(input_tensor)

class Func_f(nn.Module):
    """f_w,φ(s_t, a_t, s_t+1) = g_w(s_t, a_t) + γh_φ(s_t+1) - h_φ(s_t)"""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.reward = RewardNetwork(obs_dim, act_dim)
        self.value = RewardNetwork(obs_dim, 0)  # State-only network for potential shaping
        
    def forward(self, obs, next_obs, actions):
        
        reward_val = self.reward(obs, actions)
        value_next = self.value(next_obs, None)
        value_current = self.value(obs, None)
        
        return reward_val + 0.99 * value_next - value_current

class policy(nn.Module):
    '''
    takes the full state: obs1+obs2 concatenated and returns the probability of taking an action
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax()
        )
    
    def forward(self, state):
        return self.network(state)
        


def load_expert_data():

    with open('../../../trajectories/trajectories_20250114_110817.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    # print(trajectories['trajectories'][0])
    print("size of expert data: ", len(trajectories['trajectories']))
    return trajectories
def print_correlations(original_reward, learned_reward):
    """
    Calculate and print formatted Pearson and Spearman correlations between original and learned rewards.
    
    Parameters:
    original_reward (list): List of original rewards for each agent
    learned_reward (list): List of learned rewards for each agent
    """
    # Create a list to store results
    results = []
    
    for agent_idx in range(2):
        # Calculate correlations
        pearson_coef, pearson_p = scipy.stats.pearsonr(original_reward[agent_idx], 
                                                      learned_reward[agent_idx])
        spearman_coef, spearman_p = scipy.stats.spearmanr(original_reward[agent_idx], 
                                                         learned_reward[agent_idx])
        
        results.append({
            'Agent': f'Agent {agent_idx + 1}',
            'Correlation Type': 'Pearson',
            'Coefficient': round(pearson_coef, 3),
            'p-value': f'{pearson_p:.2e}'
        })
        
        results.append({
            'Agent': f'Agent {agent_idx + 1}',
            'Correlation Type': 'Spearman',
            'Coefficient': round(spearman_coef, 3),
            'p-value': f'{spearman_p:.2e}'
        })
    
    # Create and display DataFrame
    df = pd.DataFrame(results)
    print("\nCorrelation Analysis:")
    print(df.to_string(index=False))

def sanity_check_policy(policy_1, policy_2, expert_trajectories):
    env = gym.make("Foraging-8x8-2p-3f-v3")
    obs,_ = env.reset()

    state = torch.concatenate([torch.tensor(obs[0]), torch.tensor(obs[1])], dim=-1)
    print(state.shape)
    action_prob_1 = policy_1(state)
    action_prob_2 = policy_2(state)

    print("action_prob_1: ", action_prob_1)
    print("action_prob_2: ", action_prob_2)


def sanity_check_reward_function(reward_network_1, reward_network_2, expert_trajectories):
    env = gym.make("Foraging-8x8-2p-3f-v3")
    obs,_ = env.reset()

    actions = env.action_space.sample()
    action_1 = torch.zeros(6)
    action_1[actions[0]] = 1
    
    action_2 = torch.zeros(6)
    action_2[actions[1]] = 1

    reward_1 = reward_network_1(torch.tensor(obs[0]), action_1)
    reward_2 = reward_network_2(torch.tensor(obs[1]), action_2)
    
    print("reward_1: ", reward_1.item())
    print("reward_2: ", reward_2.item())

    original_reward = [[], []]
    learned_reward = [[], []]

    for i in range(len(expert_trajectories['trajectories'])):
        trajectory = expert_trajectories['trajectories'][i]

        for i in range(2):

            for j in trajectory['trajectories'][i]:
                obs = j['observation'][:15]
                actions = j['action'].item()
                # actions to one hot
                action = torch.zeros(6, dtype=torch.float32)
                action[actions] = 1
                original_reward[i].append(j['reward'])
                learned_reward[i].append(reward_network_1(torch.tensor(obs, dtype=torch.float32), action).item())

    print("original_reward: ", len(original_reward[0]), len(original_reward[1]))
    print("learned_reward: ", len(learned_reward[0]), len(learned_reward[1]))
    print_correlations(original_reward, learned_reward)



def sanity_check_f(f_1, f_2, expert_trajectories):
    pass



def main(obs_dim, act_dim, hidden_dim):
    trajectories = load_expert_data()
    

    print("expert trajectories loaded")

    agents = {"agent_0": {}, "agent_1": {}}
    # agents["agent_0"]["reward_network"] = RewardNetwork(obs_dim, act_dim, hidden_dim)
    # agents["agent_1"]["reward_network"] = RewardNetwork(obs_dim, act_dim, hidden_dim)
    agents["agent_0"]["f"] = Func_f(obs_dim, act_dim)
    agents["agent_1"]["f"] = Func_f(obs_dim, act_dim)
    agents["agent_0"]["policy"] = policy(state_dim=2*obs_dim, action_dim=act_dim)
    agents["agent_1"]["policy"] = policy(state_dim=2*obs_dim, action_dim=act_dim)

    # print("doing sanity check on reward functions")
    # sanity_check_reward_function(agents["agent_0"]["reward_network"], agents["agent_1"]["reward_network"], trajectories)
    
    print("doing sanity check on policy functions")
    sanity_check_policy(agents["agent_0"]["policy"], agents["agent_1"]["policy"], trajectories)
    
    print("doing sanity check on f functions")
    sanity_check_f(agents["agent_0"]["f"], agents["agent_1"]["f"], trajectories)

    num_epochs = 10
    for epoch in range(1, num_epochs+1):
        batch_size = 32
        for i in range(2):
            



if __name__ == "__main__":
    obs_dim = 15
    act_dim = 6
    main(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=256)
