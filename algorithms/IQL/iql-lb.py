import gymnasium as gym
from gymnasium.envs.registration import register
import argparse
import numpy as np
import torch.nn as nn
import torch
from collections import deque, namedtuple
import random
from torch.optim import Adam

def get_args():
    parser = argparse.ArgumentParser(description="IQL for Lbforaing envs")
    parser.add_argument("--episodes",type=int, default=100, help="number of episodes")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    return parser.parse_args()
args = get_args()

# Register the 8x8 2-player 1-food environment
s, p, f = 8, 2, 1  # size=8x8, players=2, food=1
c = False  # force_coop=False

register(
    id="Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(s, p, f, "-coop" if c else ""),
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": p,
        "min_player_level": 1,
        "max_player_level": 3,
        "min_food_level": 1,
        "max_food_level": 3,
        "field_size": (s, s),
        "max_num_food": f,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
        "normalize_reward": True,
        "grid_observation": False,
        "observe_agent_levels": True,
        "penalty": 0.0,
        "render_mode": None
    },
)

env = gym.make("Foraging-8x8-2p-1f-v3")

#####env thing done

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        # print(len(self.memory))
        if batch_size>len(self.memory):
            # print("space ledu bro")
            return None
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.memory)
    
class multiagentFCnetwork(nn.Module):
    def __init__(self, in_sizes: list[int], out_sizes: list[int]):
        super().__init__()
        activation_func = nn.ReLU

        hidden_dim = (64,64)
        n_agents = len(in_sizes)

        assert n_agents == len(out_sizes)

        self.networks = nn.ModuleList()

        for in_size, out_size in zip(in_sizes, out_sizes):
            network = [
                nn.Linear(in_size, hidden_dim[0]),
                activation_func(),
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                activation_func(),
                nn.Linear(hidden_dim[1], out_size),
            ]
            self.networks.append(nn.Sequential(*network))

    def forward(self, inputs: list[torch.Tensor]):
        futures = [
            torch.jit.fork(model, inputs[i]) for i, model in enumerate(self.networks)
        ]
        results = [torch.jit.wait(fut) for fut in futures]

        return results


num_episodes = args.episodes
obs_sizes = (9,9)
action_sizes = (6, 6)

Q_networks = multiagentFCnetwork(obs_sizes, action_sizes)
target_Q_networks = multiagentFCnetwork(obs_sizes, action_sizes)

for network1, network2 in zip(Q_networks.networks, target_Q_networks.networks):
    network2.load_state_dict(network1.state_dict())

optimizers = [Adam(network.parameters(), lr=args.lr) for network in Q_networks.networks]

# check if both the networks state_dict is the same
# print(Q_networks.networks[0].state_dict())
# assert_networks_equal(Q_networks, target_Q_networks, index=0)
# assert Q_networks.networks[0].state_dict() == target_Q_networks.networks[0].state_dict()

agent_wise_replay_buffer = [ReplayBuffer() for _ in range(p)]
time_total = 0

for ep in range(num_episodes):
    obs,_ = env.reset()
    agent_wise_rewards = np.array([0.,0.])
    count_steps = 0
    episode_loss = 0
    while True:
        actions = env.action_space.sample()
        next_states, rewards, done, turnc, _ = env.step(actions)
        agent_wise_rewards += np.array(rewards)
        count_steps+=1

        #store the transitions.
        for i in range(len(actions)):
            agent_wise_replay_buffer[i].push(
                obs[i],
                actions[i],
                rewards[i],
                next_states[i],
                done,
            )

        obs = next_states

        #update the network
        flag = True
        
        #sample from the replay buffer
        for i in range(len(actions)): # for each agent.
            batch_size = 32
            
            trajectories = agent_wise_replay_buffer[i].sample(batch_size=batch_size)

            if trajectories is not None:
                state_tensor = torch.tensor(np.array(trajectories.state))
                action_tensor = torch.tensor(trajectories.action)
                next_state_tensor = torch.tensor(np.array(trajectories.next_state))
                dones_tensor = torch.tensor(trajectories.done)
                reward_tensor = torch.tensor(trajectories.reward)
                #print(state_tensor.shape, action_tensor.shape, next_state_tensor.shape, dones_tensor.shape, reward_tensor.shape)
            
                q_learning_values = Q_networks.networks[i](state_tensor)
                target_q_values = target_Q_networks.networks[i](next_state_tensor)
                # optimizers[i].zero_grad()
                action_indices = action_tensor.view(batch_size, -1)
                current_q_values = q_learning_values.gather(1, action_indices)

                # print("current_q_values ", current_q_values)
                # Compute target Q-values using Bellman equation
                with torch.no_grad():
                    # Get maximum Q-value for next states
                    next_q_values, _ = target_q_values.max(dim=1, keepdim=True)
                    
                    # Compute target using Bellman equation
                    # Q_target = reward + gamma * max(Q(s', a')) * (1 - done)
                    gamma = 0.99  # discount factor
                    expected_q_values = reward_tensor.view(-1, 1) + gamma * next_q_values 
                    # print(expected_q_values.shape)
                # Compute loss (typically MSE or Huber loss)
                loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values)

                # Backpropagate and update
                loss.backward()
                optimizers[i].step()

                episode_loss+=loss.item()

                # Update target network
                if time_total % 100 == 0:
                    for target_param, param in zip(target_Q_networks.networks[i].parameters(), Q_networks.networks[i].parameters()):
                        target_param.data.copy_(param.data)
                    # assert_networks_equal(Q_networks, target_Q_networks, index=i)
                
                time_total+=1

        if done or turnc:
            break

    print(f"Episode:{ep+1}, agent_wise_rewards:{agent_wise_rewards}, number of steps: {count_steps}, done:{done}, truncated:{turnc}, avg episode loss, {episode_loss/count_steps}")
    
#save the q-networks
i = 0
for network in Q_networks.networks:
    torch.save(network.state_dict(), f"networks/q_networks_{i}.pth")
    i+=1

print("done, networks saved")