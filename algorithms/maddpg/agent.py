import torch
# import torch.nn.functional as F
# import numpy as np
from networks import ActorNetwork, CriticNetwork

class DDPGAgent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, 
                 chkpt_dir='tmp/maddpg/',
                 alpha=0.01, beta=0.01, gamma=0.95,
                 tau=0.01, fc1=64, fc2=64):
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'
        self.actor = ActorNetwork(actor_dims, n_actions, fc1, fc2)
        print("critic dim in agent file: ", critic_dims, n_actions*n_agents, fc1, fc2)
        self.critic = CriticNetwork(critic_dims, n_actions*n_agents, fc1, fc2)
        self.target_actor = ActorNetwork(actor_dims, n_actions, fc1, fc2)
        self.target_critic = CriticNetwork(critic_dims, n_actions*n_agents, fc1, fc2)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=beta)
        
        self.update_network_parameters(tau=1)
        
        self.chkpt_dir = chkpt_dir

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                   (1-tau)*target_actor_dict[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                    (1-tau)*target_critic_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation, explore=True):
        state = torch.tensor([observation], dtype=torch.float)
        actions = self.actor.forward(state)
        if explore:
            noise = torch.randn(self.n_actions)
            actions = torch.clamp(actions + noise, 0.0, 1.0)
        return actions.detach().numpy()[0]

    def save_models(self):
        torch.save(self.actor.state_dict(), 
                  f'{self.chkpt_dir}/{self.agent_name}_actor.pth')
        torch.save(self.critic.state_dict(), 
                  f'{self.chkpt_dir}/{self.agent_name}_critic.pth')
        torch.save(self.target_actor.state_dict(),
                  f'{self.chkpt_dir}/{self.agent_name}_target_actor.pth')
        torch.save(self.target_critic.state_dict(),
                  f'{self.chkpt_dir}/{self.agent_name}_target_critic.pth')

    def load_models(self):
        self.actor.load_state_dict(
            torch.load(f'{self.chkpt_dir}/{self.agent_name}_actor.pth'))
        self.critic.load_state_dict(
            torch.load(f'{self.chkpt_dir}/{self.agent_name}_critic.pth'))
        self.target_actor.load_state_dict(
            torch.load(f'{self.chkpt_dir}/{self.agent_name}_target_actor.pth'))
        self.target_critic.load_state_dict(
            torch.load(f'{self.chkpt_dir}/{self.agent_name}_target_critic.pth'))
