import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
from algorithms.random_policy import random_policy

# Set up argument parser
parser = argparse.ArgumentParser(description="Run random policy in Foraging environment")
parser.add_argument("--collect_video", action="store_true", help="Flag to collect video of the run")
parser.add_argument("--visualize", action="store_true", help="Flag to visualize the environment")
parser.add_argument("--ignore_warnings", action="store_true", help="Flag to ignore warnings")
# Register the 8x8 2-player 1-food environment
parser.add_argument("--size", type=int, default=8, help="Size of the environment (NxN)")
parser.add_argument("--players", type=int, default=2, help="Number of players")
parser.add_argument("--food", type=int, default=1, help="Number of food items")
parser.add_argument("--force_coop", action="store_true", help="Flag to force cooperation")

args = parser.parse_args()

if args.ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

s, p, f = args.size, args.players, args.food
c = args.force_coop
collect_video = args.collect_video
visualize = args.visualize
# print(collect_video, visualize)

print("\nNote that video collection is not wokring, it needs to be fixed.\n")

# render_mode = "rgb_array" if collect_video else "human" if visualize else None
 
env_name = "Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(s, p, f, "-coop" if c else "")

if env_name not in gym.envs.registry.keys():
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
        },
    )


env = gym.make(env_name)

random_policy(env, 30, visualize=args.visualize)
