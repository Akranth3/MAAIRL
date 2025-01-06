from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from datetime import datetime
import pickle

@dataclass
class Step:
    """Single step in a trajectory"""
    state: Any
    action: Any
    reward: float
    # timestamp: datetime = field(default_factory=datetime.now)

class Trajectory:
    """Stores a single trajectory for one agent"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.steps: List[Step] = []
        self.total_reward = 0.0
        
    def add_step(self, state: Any, action: Any, reward: float) -> None:
        """Add a step to the trajectory"""
        self.steps.append(Step(state, action, reward))
        self.total_reward += reward
        
    def get_states(self) -> List[Any]:
        """Get list of all states"""
        return [step.state for step in self.steps]
    
    def get_actions(self) -> List[Any]:
        """Get list of all actions"""
        return [step.action for step in self.steps]
    
    def get_rewards(self) -> List[float]:
        """Get list of all rewards"""
        return [step.reward for step in self.steps]

class TrajectoryRecorder:
    """Records trajectories for multiple agents across episodes"""
    def __init__(self):
        self.episodes: Dict[int, Dict[str, Trajectory]] = {}
        self.current_episode = 0
        
    def start_episode(self) -> None:
        """Start recording a new episode"""
        self.current_episode += 1
        self.episodes[self.current_episode] = {}
        
    def add_step(self, episode: int, agent_name: str, 
                state: Any, action: Any, reward: float) -> None:
        """Record a step for an agent"""
        if episode not in self.episodes:
            self.episodes[episode] = {}
            
        if agent_name not in self.episodes[episode]:
            self.episodes[episode][agent_name] = Trajectory(agent_name)
            
        self.episodes[episode][agent_name].add_step(state, action, reward)
        
    def get_episode_trajectories(self, episode: int) -> Dict[str, Trajectory]:
        """Get all trajectories for a specific episode"""
        return self.episodes.get(episode, {})
    
    def get_agent_trajectory(self, episode: int, agent_name: str) -> Optional[Trajectory]:
        """Get trajectory for specific agent in specific episode"""
        return self.episodes.get(episode, {}).get(agent_name)
    
    def save_trajectories(self, filepath: str) -> None:
        """Save trajectories to file"""
        data = {
            str(episode): {
                agent: {
                    'states': traj.get_states(),
                    'actions': traj.get_actions(),
                    'rewards': traj.get_rewards(),
                }
                for agent, traj in agent_trajs.items()
            }
            for episode, agent_trajs in self.episodes.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
    @staticmethod
    def load_trajectories(filepath: str) -> 'TrajectoryRecorder':
        """Load trajectories from file"""
        recorder = TrajectoryRecorder()
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for episode_str, agent_trajs in data.items():
            episode = int(episode_str)
            for agent_name, traj_data in agent_trajs.items():
                for state, action, reward in zip(
                    traj_data['states'],
                    traj_data['actions'],
                    traj_data['rewards']
                ):
                    recorder.add_step(episode, agent_name, state, action, reward)
                    
        return recorder
