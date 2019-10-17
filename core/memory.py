import random
import numpy as np


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        """ Store new transition. """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Sample randomly. """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self):
        return len(self.memory)
