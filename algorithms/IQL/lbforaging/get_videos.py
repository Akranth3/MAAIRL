import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from algorithms.IQL.lbforaging.iql import MultiAgentFCNetwork
import lbforaging
import time
import numpy as np


#load the models
agents = MultiAgentFCNetwork(in_sizes=[9,9], out_sizes=[6,6])

i = 0
for network in agents.networks:
    network.load_state_dict(torch.load(f"networks/q_network_{i}_final_30000.pth"))
    i+=1

num_episodes = 300


def select_action(states, agents = agents):
    actions = []
    i = 0
    with torch.no_grad():
        for q_network in agents.networks:
            temp_state = torch.tensor(states[i], dtype=torch.float32)
            q_values = q_network(temp_state)
            action = torch.argmax(q_values).item()
            actions.append(action)
            i+=1
    return actions

env = gym.make("Foraging-8x8-2p-1f-v3")
# env = gym.wrappers.RecordVideo(env=env, video_folder="videos", name_prefix="test-video")

success_episodes = 0
for i in range(num_episodes):
    states,_ = env.reset()
    count = 0
    rewards = 0
    agent_1_actions = []
    agent_2_actions = []

    while True:
        actions = select_action(states)
        agent_1_actions.append(actions[0])
        agent_2_actions.append(actions[1])
        # print(actions)
        next_states, rewards, dones, _, _ = env.step(actions)
        states = next_states
        # img = env.render()
        count += 1
        # time.sleep(0.1)
        if dones:
            break
    variance_action_1 = np.var(agent_1_actions)
    variance_action_2 = np.var(agent_2_actions)
    if count<50:
        success_episodes+=1
    print(f"episode {i} completed in {count} steps, variance of actions: {variance_action_1:.2f}, {variance_action_2:.2f}")

# env.close_video_recorder()
print(f"Success rate: {success_episodes*100/num_episodes}")

print("Models loaded successfully")