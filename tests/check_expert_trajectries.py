import pickle
import numpy as np
from utils.expert_trajectories import TrajectoryRecorder

def print_separator():
    print("\n" + "="*50 + "\n")

def inspect_trajectories():
    # Load data
    with open('../demonstration/maddpg/simple_adversary_expert_trajectories.pkl', 'rb') as f:
        recorder = pickle.load(f)
    
    # Basic structure inspection
    print("1. Basic Data Structure")
    print("-" * 20)
    print("Number of episodes:", len(recorder.episodes))
    print("Episode numbers:", sorted(recorder.episodes.keys()))
    print_separator()

    # Detailed inspection of first episode
    print("2. First Episode Details")
    print("-" * 20)
    first_episode = min(recorder.episodes.keys())
    episode_data = recorder.episodes[first_episode]
    
    print(f"Episode {first_episode}:")
    print("Agents in episode:", list(episode_data.keys()))
    print("Number of agents:", len(episode_data))
    print_separator()

    # Inspect one agent's trajectory
    print("3. Single Agent Trajectory")
    print("-" * 20)
    first_agent = list(episode_data.keys())[0]
    agent_traj = episode_data[first_agent]
    
    print(f"Agent: {first_agent}")
    print("Number of steps:", len(agent_traj.steps))
    
    if len(agent_traj.steps) > 0:
        first_step = agent_traj.steps[0]
        print("\nFirst step details:")
        print(f"State shape: {np.array(first_step.state).shape}")
        print(f"Action shape: {np.array(first_step.action).shape if first_step.action is not None else 'None'}")
        print(f"Reward: {first_step.reward}")
        # print(f"Timestamp: {first_step.timestamp}")
        
    print_separator()

    # Statistical analysis
    print("4. Basic Statistics")
    print("-" * 20)
    
    # Get episode lengths
    episode_lengths = {ep: len(list(recorder.episodes[ep].values())[0].steps) 
                      for ep in recorder.episodes.keys()}
    
    print("Episode length statistics:")
    print(f"Mean: {np.mean(list(episode_lengths.values())):.2f}")
    print(f"Min: {min(episode_lengths.values())}")
    print(f"Max: {max(episode_lengths.values())}")
    print_separator()

    # Detailed look at trajectories
    print("5. Detailed Data Sample")
    print("-" * 20)
    sample_traj = episode_data[first_agent]
    
    if len(sample_traj.steps) > 0:
        first_step = sample_traj.steps[0]
        print("First state:", np.array(first_step.state))
        print("First action:", np.array(first_step.action) if first_step.action is not None else None)
        print("First reward:", first_step.reward)
    print_separator()

    # Reward statistics per agent
    print("6. Reward Statistics Per Agent")
    print("-" * 20)
    
    for agent in episode_data.keys():
        trajectory = episode_data[agent]
        rewards = [step.reward for step in trajectory.steps]
        
        print(f"\nAgent {agent}:")
        print(f"Total reward: {sum(rewards):.2f}")
        print(f"Mean reward: {np.mean(rewards):.2f}")
        print(f"Min reward: {min(rewards):.2f}")
        print(f"Max reward: {max(rewards):.2f}")
    print_separator()

    # Optional: Plot reward distribution
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        for agent in episode_data.keys():
            trajectory = episode_data[agent]
            rewards = [step.reward for step in trajectory.steps]
            plt.plot(rewards, label=f'Agent {agent}')
        
        plt.title('Reward Distribution Over Time (First Episode)')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    inspect_trajectories()