import sys
sys.path.append('../../')

from pettingzoo.mpe import simple_adversary_v3
import numpy as np
from maddpg import MADDPG
import os
from utils.model_short_summary import summary_model
import argparse


def evaluate(maddpg_agents, env, n_episodes=100):
    """
    Evaluate MADDPG agents in the given environment.
    
    Args:
        maddpg_agents: MADDPG agents object
        env: PettingZoo environment
        n_episodes: Number of episodes to evaluate
        
    Returns:
        float: Mean score across all episodes
    """
    scores = []
    
    for episode in range(n_episodes):
        env.reset()
        score = 0
        
        for agent in env.agent_iter():
            observation, reward, done, truncated, info = env.last()
            
            if done or truncated:
                action = None
            else:
                action = maddpg_agents.choose_actions(observation, agent, explore=False)
            # print(action)
            env.step(action)
            score += reward
            
            if done or truncated:
                break
        
        scores.append(score)
    
    return np.mean(scores)

def train_maddpg(args=None):
    env = simple_adversary_v3.env(continuous_actions=True)
    env.reset()
    n_agents = env.num_agents
    
    obs_dims = [env.observation_space(agent).shape[0] for agent in env.agents]
    action_dims = [env.action_space(agent).shape[0] for agent in env.agents]

    print("Observation dimensions:", obs_dims)
    print("Action dimensions:", action_dims)
    print("Number of agents:", n_agents)

    maddpg_agents = MADDPG(scenario_name='simple_adversary',
                          n_agents=n_agents,
                          actor_dims=obs_dims,
                          critic_dims=obs_dims,
                          action_dims=action_dims)
    
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print("MADDPG Agents")
    # for agent in maddpg_agents.agents:
    #     # print(agent)
    #     print("Actor Network")
    #     summary_model(agent.actor)

    #     print("Critic Network")
    #     summary_model(agent.critic)
    
    # print("the input to the critics is all the observations flattened")
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")


    
    
    n_episodes = args.n_episodes
    max_steps = 25
    batch_size = 1024
    
    best_score = float('-inf')
    score_history = []
    
    for episode in range(n_episodes):
        env.reset()
        score = 0
        
        for agent in env.agent_iter():
            observation, reward, done, truncated, info = env.last()
            
            if done or truncated:
                action = None
            else:
                # print("Observation: ", observation)
                # print("agent: ", agent)
                # print("------")
                action = maddpg_agents.choose_actions(observation, agent)
                # print("------")
            # print(action)

            # print("turnacation: ", truncated)
            # print("info: ", info)
            env.step(action)
            if not done:
                next_observation = env.observe(agent)
                # print("\nDebugging the buffer")
                # print("Observation: ", observation)
                # print("Action: ", action)
                # print("Reward: ", reward)
                # print("Next Observation: ", next_observation)
                # print("Done: ", done)
                maddpg_agents.memory.store_transition(agent, observation, action,
                                                    reward, next_observation,
                                                    done)
                
            score += reward
            
            if done or truncated:
                break
                
        maddpg_agents.learn(batch_size)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            maddpg_agents.save_checkpoint()
            
        if episode % 100 == 0:
            print(f'Episode {episode}, Average Score: {avg_score:.2f}')
            
    return maddpg_agents

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_episodes', type=int, default=500)

    os.makedirs('tmp/maddpg', exist_ok=True)
    maddpg_agents = train_maddpg(args=argparser.parse_args())
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("Traingin done using MADDPG")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    
    
    # Evaluate the trained agents
    env = simple_adversary_v3.env(continuous_actions=True)
    mean_score = evaluate(maddpg_agents, env)
    print(f'Mean evaluation score: {mean_score:.2f}')