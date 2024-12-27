import torch
import numpy as np
from algorithms.maairl.maairl import MAAIRL
from algorithms.maairl.utils import ReplayBuffer
import gymnasium
# import make_env  # Your multi-particle environment
from pettingzoo.mpe import simple_adversary_v3


def train_maairl(env_name, num_episodes=10000, batch_size=64):
    # Initialize environment
    env = simple_adversary_v3.env(continuous_actions=True)
    env.reset()

    num_agents = env.num_agents
    
    # Get state and action dimensions for each agent
    state_dims = [env.observation_space(i).shape[0] for i in env.agents]
    action_dims = [env.action_space(i).shape[0] for i in env.agents]
    
    # Initialize MAAIRL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maairl = MAAIRL(num_agents, state_dims, action_dims, device)
    
    # Initialize replay buffers
    expert_buffer = ReplayBuffer(1e6)
    policy_buffer = ReplayBuffer(1e6)

    print("basic thing is working ig.")
    
    # Load expert demonstrations (you need to implement this based on your data)
    load_expert_demonstrations(expert_buffer)
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert states to tensor
            states_tensor = [torch.FloatTensor(state).to(device) for state in states]
            
            # Get actions from policy
            actions, _ = maairl.get_actions(states_tensor)
            actions_np = [action.detach().cpu().numpy() for action in actions]
            
            # Take step in environment
            next_states, rewards, done, _ = env.step(actions_np)
            
            # Store transition in buffer
            policy_buffer.add(states, actions_np, next_states, rewards, done)
            
            states = next_states
            episode_reward += np.mean(rewards)
            
            # Update networks if enough data is collected
            if len(policy_buffer) > batch_size:
                # Sample from buffers
                expert_batch = expert_buffer.sample(batch_size)
                policy_batch = policy_buffer.sample(batch_size)
                
                # Update discriminator
                disc_loss = maairl.update_discriminator(
                    expert_batch['states'], expert_batch['actions'], expert_batch['next_states'],
                    policy_batch['states'], policy_batch['actions'], policy_batch['next_states']
                )
                
                # Update policy
                policy_loss = maairl.update_policy(
                    policy_batch['states'], 
                    policy_batch['actions'],
                    policy_batch['next_states']
                )
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward}")
            # Save model periodically
            maairl.save(f"models/maairl_{env_name}_{episode}.pt")

if __name__ == "__main__":
    train_maairl("simple_spread")  # or whatever your environment name is