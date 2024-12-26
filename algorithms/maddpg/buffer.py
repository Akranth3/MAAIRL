import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, num_agents, obs_dims, action_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.num_agents = num_agents
        
        self.state_memory = []
        self.new_state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.done_memory = []
        
        for i in range(num_agents):
            self.state_memory.append(np.zeros((max_size, obs_dims[i])))
            self.new_state_memory.append(np.zeros((max_size, obs_dims[i])))
            self.action_memory.append(np.zeros((max_size, action_dims[i])))
            self.reward_memory.append(np.zeros(max_size))
            self.done_memory.append(np.zeros(max_size, dtype=bool))
        
        print("state_memory: ", len(self.state_memory))
        print("new_state_memory: ", len(self.new_state_memory))
        print("action_memory: ", len(self.action_memory))
        print("reward_memory: ", len(self.reward_memory))
        print("done_memory: ", len(self.done_memory))

        print("state_memory: ", self.state_memory[0].shape, self.state_memory[1].shape, self.state_memory[2].shape)
        print("new_state_memory: ", self.new_state_memory[0].shape, self.new_state_memory[1].shape, self.new_state_memory[2].shape)
        print("action_memory: ", self.action_memory[0].shape, self.action_memory[1].shape, self.action_memory[2].shape)
        print("reward_memory: ", self.reward_memory[0].shape, self.reward_memory[1].shape, self.reward_memory[2].shape)
        print("done_memory: ", self.done_memory[0].shape, self.done_memory[1].shape, self.done_memory[2].shape)

    def store_transition(self, agent_name, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        # print("agent_name: ", agent_name)
        name_to_idx = {'adversary_0': 0, 'agent_0': 1, 'agent_1': 2}
        agent_idx = name_to_idx[agent_name]
        # print("agent_idx: ", agent_idx)
        # print(self.state_memory[agent_idx].shape)
        # print("whats wrong with index: ", index)
        # for agent_idx in range(self.num_agents):
        self.state_memory[agent_idx][index] = state
        self.new_state_memory[agent_idx][index] = next_state
        self.action_memory[agent_idx][index] = action
        self.reward_memory[agent_idx][index] = reward
        self.done_memory[agent_idx][index] = done

        if agent_idx == self.num_agents - 1:
            self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = [self.state_memory[i][batch] for i in range(self.num_agents)]
        actions = [self.action_memory[i][batch] for i in range(self.num_agents)]
        rewards = [self.reward_memory[i][batch] for i in range(self.num_agents)]
        next_states = [self.new_state_memory[i][batch] for i in range(self.num_agents)]
        dones = [self.done_memory[i][batch] for i in range(self.num_agents)]

        return states, actions, rewards, next_states, dones