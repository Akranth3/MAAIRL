import gymnasium as gym
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Check if a Gym environment is registered.')
parser.add_argument('env_name', type=str, help='Name of the Gym environment to check')

# Parse arguments
args = parser.parse_args()

# Check if the environment is registered
envs = gym.envs.registry.keys()

if args.env_name in envs:
    print(f"Environment '{args.env_name}' is registered.")
else:
    print("Environment is not registered, you are in trouble.")

