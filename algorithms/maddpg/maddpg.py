import torch
import numpy as np
from algorithms.maddpg.agent import DDPGAgent
from algorithms.maddpg.buffer import ReplayBuffer
import torch.nn.functional as F

class MADDPG:
    def __init__(self, scenario_name, n_agents, actor_dims, critic_dims, 
                 action_dims, gamma=0.99, tau=0.01, lr_actor=0.01, 
                 lr_critic=0.01, buffer_size=1000000):
        
        self.agents = []
        self.n_agents = n_agents
        self.scenario_name = scenario_name
        
        for agent_idx in range(self.n_agents):
            self.agents.append(DDPGAgent(actor_dims[agent_idx], critic_dims,
                                       action_dims[agent_idx], n_agents, agent_idx,
                                       gamma=gamma, tau=tau,
                                       alpha=lr_actor, beta=lr_critic))
        
        self.memory = ReplayBuffer(buffer_size, n_agents, actor_dims, action_dims)
        
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_actions(self, raw_obs, agent_name, explore=True):
        map_agent_name_to_idx = {'adversary_0': 0, 'agent_0': 1, 'agent_1':2}
        needed_agent_idx = map_agent_name_to_idx[agent_name]
        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == needed_agent_idx:
                # print(agent_idx, agent_name)
                action = agent.choose_action(raw_obs, explore)
            # action = agent.choose_action(raw_obs[agent_idx], explore)
            # actions.append(action)
        return action

    def learn(self, batch_size):
        if self.memory.mem_cntr < batch_size:
            return
        
        states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        states = [torch.tensor(state, dtype=torch.float, device=device) for state in states]
        actions = [torch.tensor(action, dtype=torch.float, device=device) for action in actions]
        rewards = [torch.tensor(reward, dtype=torch.float, device=device) for reward in rewards]
        next_states = [torch.tensor(next_state, dtype=torch.float, device=device) for next_state in next_states]
        dones = [torch.tensor(done, dtype=torch.bool, device=device) for done in dones]
        
        all_target_actions = []
        for agent_idx, agent in enumerate(self.agents):
            target_actions = agent.target_actor.forward(next_states[agent_idx])
            all_target_actions.append(target_actions)
            
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(next_states, all_target_actions)
            critic_value = agent.critic.forward(states, actions)
            
            target = rewards[agent_idx].view(-1, 1) + agent.gamma * \
                        critic_value_ * (1 - dones[agent_idx].float().view(-1, 1))

            # target = rewards[agent_idx].view(-1, 1) + agent.gamma * \
            #         critic_value_ * (1 - dones[agent_idx].view(-1, 1))
            
            critic_loss = F.mse_loss(critic_value, target.detach())
            agent.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic_optimizer.step()
            
            actions_for_actor = actions.copy()
            actions_for_actor[agent_idx] = agent.actor.forward(states[agent_idx])
            actor_loss = -agent.critic.forward(states, actions_for_actor).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor_optimizer.step()
            
            agent.update_network_parameters()