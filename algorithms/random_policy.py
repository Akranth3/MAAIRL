import gymnasium as gym
import numpy as np
import time
from gymnasium.wrappers import RecordVideo


def random_policy(env, max_steps=30, visualize=False, collect_video=False, video_folder='logger/videos'):
    if collect_video:
        env = RecordVideo(env, video_folder=video_folder)
    
    obs, _ = env.reset()
    sleep = 0.2
    rewards = []
    # store_vals = []
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if visualize:
            val = env.render()
            # store_vals.append(val)
            time.sleep(sleep)
        if done:
            break

    # print("the type of vals is: ", type(store_vals), type(store_vals[0]))
    if collect_video:
        env.close()
    
    return rewards

