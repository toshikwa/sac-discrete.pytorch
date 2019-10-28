from collections import deque
import random
import numpy as np
import torch


class ReplayMemory:

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """ Store new transition. """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """ Sample randomly. """
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones =\
            map(np.stack, zip(*experiences))

        # current state: (num_batches, 4, 84, 84)
        states = torch.FloatTensor(states).to(self.device)
        # next state: (num_batches, 4, 84, 84)
        next_states = torch.FloatTensor(next_states).to(self.device)
        # action: (num_batches, 1)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        # reward: (num_batches, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # done: (num_batches, 1)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
