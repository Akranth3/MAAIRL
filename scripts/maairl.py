# import pickle


# '''
# The environment is a simple predator-prey environment with two agents. The predator is trying to catch the prey.
# the action space is continuous.
# '''

# with open('../demonstration/maddpg/simple_adversary_expert_trajectories.pkl', 'rb') as f:
#     recorder = pickle.load(f)

# trajectories = []

# count = 0
# for episode in recorder.episodes:
#     count+=1
#     episode_data = recorder.episodes[episode]
#     dict = {}
#     for agent in episode_data:
#         agent_traj = episode_data[agent]
#         steps = []
#         for step in agent_traj.steps:
#             steps.append((step.state, step.action, step.reward))
#         dict[agent] = steps
#     trajectories.append(dict)

# print("Number of episodes:", count)

# print(trajectories[0]['agent_1'][0])
# # {"agent1": [(state, action, reward), (state, action, reward), ...], "agent2": ...}

# # got expert trajectories

# # multi agent adverserial inverse Reinforcement learning.


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pickle

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RewardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.final = nn.Linear(128, action_dim)
        self.sigmoid = nn.Sigmoid()
        # self.mean = nn.Linear(128, action_dim)
        # self.log_std = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = self.network(state)
        # mean = self.mean(x)
        # log_std = self.log_std(x)
        # std = torch.exp(log_std)
        x = self.final(x)
        x = self.sigmoid(x)
        return x

class MAIRL:
    def __init__(self, state_dims, action_dim, num_agents, learning_rate=3e-4):
        self.state_dims = state_dims
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Initialize reward networks for each agent
        self.reward_nets = [RewardNetwork(state_dims[i], action_dim) for i in range(num_agents)]
        self.reward_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) 
                                for net in self.reward_nets]
        
        # Initialize policy networks for each agent
        self.policy_nets = [PolicyNetwork(state_dims[i], action_dim) for i in range(num_agents)]
        self.policy_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) 
                                for net in self.policy_nets]
        
    def compute_reward(self, states, actions, agent_idx):
        """Compute the reward for a given agent's states and actions."""
        return self.reward_nets[agent_idx](states, actions)
    
    def sample_action(self, state, agent_idx):
        """Sample an action from the policy network for a given agent."""
        action = self.policy_nets[agent_idx](state)
        # dist = Normal(mean, std)
        # action = dist.sample()
        # log_prob = dist.log_prob(action)
        return action
    
    def update_reward(self, expert_states, expert_actions, policy_states, 
                     policy_actions, agent_idx):
        """Update reward network using the AIRL objective."""
        expert_rewards = self.compute_reward(expert_states, expert_actions, agent_idx)
        policy_rewards = self.compute_reward(policy_states, policy_actions, agent_idx)
        
        # Compute loss using AIRL objective
        expert_labels = torch.ones_like(expert_rewards)
        policy_labels = torch.zeros_like(policy_rewards)
        
        loss = nn.BCEWithLogitsLoss()(torch.cat([expert_rewards, policy_rewards], dim=0),
                                     torch.cat([expert_labels, policy_labels], dim=0))
        
        self.reward_optimizers[agent_idx].zero_grad()
        loss.backward()
        self.reward_optimizers[agent_idx].step()
        
        return loss.item()
    
    def update_policy(self, states, agent_idx):
        """Update policy network using the current reward function."""
        actions, log_probs = self.sample_action(states, agent_idx)
        rewards = self.compute_reward(states, actions, agent_idx)
        
        # Policy gradient loss
        policy_loss = -(log_probs * rewards).mean()
        
        self.policy_optimizers[agent_idx].zero_grad()
        policy_loss.backward()
        self.policy_optimizers[agent_idx].step()
        
        return policy_loss.item()

def process_trajectories(trajectory_path):
    """Process the expert trajectories from pickle file."""
    with open(trajectory_path, 'rb') as f:
        recorder = pickle.load(f)
    
    trajectories = []
    for episode in recorder.episodes:
        episode_data = recorder.episodes[episode]
        dict = {}
        for agent in episode_data:
            agent_traj = episode_data[agent]
            steps = []
            for step in agent_traj.steps:
                steps.append((step.state, step.action, step.reward))
            dict[agent] = steps
        trajectories.append(dict)
    print(len(trajectories[0]))
    print(trajectories[0].keys())
    print(trajectories[0]['agent_1'][0])
    return trajectories

def train_mairl(env_path, num_epochs=1000, batch_size=64):

    # Load expert trajectories
    
    trajectories = process_trajectories(env_path)
    print("this thing workss")

    # Initialize MAIRL
    state_dims = []
    for agent in trajectories[0]:
        state_dims.append(len(trajectories[0][agent][0][0]))
    
    action_dim = len(trajectories[0]['agent_1'][0][1])  # Assuming all actions have same dim
    num_agents = len(trajectories[0])

    print("The state action and number of agents respectively are these: "  ,state_dims, action_dim, num_agents)

    
    mairl = MAIRL(state_dims, action_dim, num_agents)


    print("analyzing the mairl object")
    # print("################################################")
    # print("below are the reward nets.")
    # print(mairl.reward_nets)
    # print("################################################")
    # print("below are the policy nets.")
    # print(mairl.policy_nets)
    print("the output is mean and std deviation of the actions, 5, 5 both vectors? in ddpg it's just a 5 vector?")
    print("changed the policy network similar to the MADDPG policy network (actor).")


    agent_names = ['adversary_0', 'agent_0', 'agent_1']
    for epoch in range(num_epochs):
        # Sample batch of trajectories
        batch_indices = np.random.choice(len(trajectories), batch_size)
        print("batch_indices", batch_indices)
        for agent_idx in range(num_agents):
            agent_name = agent_names[agent_idx]
            # print("agent_name", agent_name)
            # Collect states and actions
            # expert_states = []
            # expert_actions = []
            # for idx in batch_indices:
            #     traj = trajectories[idx][agent_name]
            #     states, actions, _ = zip(*traj)
            #     if actions is not None:
            #         expert_states.extend(states)
            #         expert_actions.extend(actions)
            
            # print(type(expert_states))
            # print(type(expert_states[10]))

            # print(expert_actions)

            # expert_states_np = np.array(expert_states, dtype=np.float32)
            # expert_actions_np = np.array(expert_actions, dtype=np.float32)
            
            # print(expert_actions_np.shape, expert_states_np.shape)
            # # Convert to PyTorch tensors
            # expert_states = torch.FloatTensor(expert_states_np)
            # expert_actions = torch.FloatTensor(expert_actions_np)
            expert_states = []
            expert_actions = []
            
            for idx in batch_indices:
                traj = trajectories[idx][agent_name]
                states, actions, _ = zip(*traj)
                
                # Skip if actions is None
                if actions is None:
                    continue
                    
                # Debug info
                print(f"\nTrajectory {idx} info:")
                print(f"Number of states: {len(states)}")
                print(f"Number of actions: {len(actions)}")
                if len(actions) > 0:
                    print(f"Sample action type: {type(actions[0])}")
                    print(f"Sample action: {actions[0]}")
                
                # Process states and actions
                for state, action in zip(states, actions):
                    if action is not None:  # Skip None actions
                        # Convert action to numpy array if it isn't already
                        if not isinstance(action, np.ndarray):
                            try:
                                action = np.array(action, dtype=np.float32)
                            except Exception as e:
                                print(f"Failed to convert action to numpy array: {action}")
                                print(f"Error: {e}")
                                continue
                        
                        # Ensure action is 1D
                        action = action.flatten()
                        
                        expert_states.append(state)
                        expert_actions.append(action)
            
            if not expert_actions:  # Check if we have any valid actions
                raise ValueError("No valid actions found in trajectories")
            
            # Get the expected action dimension from the first action
            action_dim = len(expert_actions[0])
            
            # Verify all actions have the same dimension
            valid_states = []
            valid_actions = []
            for i, (state, action) in enumerate(zip(expert_states, expert_actions)):
                if len(action) == action_dim:
                    valid_states.append(state)
                    valid_actions.append(action)
                else:
                    print(f"Skipping inconsistent action at index {i}. "
                        f"Expected dim: {action_dim}, Got: {len(action)}")
            
            # Convert to numpy arrays
            expert_states_np = np.array(valid_states, dtype=np.float32)
            expert_actions_np = np.stack(valid_actions).astype(np.float32)
            
            
            # Convert to PyTorch tensors
            expert_states = torch.FloatTensor(expert_states_np)
            expert_actions = torch.FloatTensor(expert_actions_np)

            print("expert states and actions are converted to torch tensors.")
            print("expert states shape", expert_states.shape)
            print("expert actions shape", expert_actions.shape)
            print("things are smooth")
            
            # Generate policy trajectories
            policy_states = expert_states.clone()  # Using expert states for simplicity
            '''this looks shady???????'''



            

            print("stopped at the policy loss, i need to understand the paper first otherwise i can't see where things are going correct or wrong.")
            
            
            
            
            
            
            
            
            policy_actions = mairl.sample_action(policy_states, agent_idx)
            
            # Update reward and policy
            reward_loss = mairl.update_reward(expert_states, expert_actions, 
                                            policy_states, policy_actions, agent_idx)
            policy_loss = mairl.update_policy(expert_states, agent_idx)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Agent {agent_name}")
                print(f"Reward Loss: {reward_loss:.4f}, Policy Loss: {policy_loss:.4f}")
    
    return mairl

if __name__ == "__main__":
    env_path = '../demonstration/maddpg/simple_adversary_expert_trajectories.pkl'
    trained_model = train_mairl(env_path)