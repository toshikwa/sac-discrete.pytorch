import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler


class PER(dict):
    keys = ["state", "action", "reward", "next_state", "done", "priority"]

    def __init__(self, capacity, state_shape, action_shape, device, alpha=0.6,
                 beta=0.4, beta_annealing=0.001, epsilon=1e-4):
        super(PER, self).__init__()
        # memory params
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        # per params
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

        self._n = 0
        self._p = 0

        self["state"] = np.zeros(
            (self.capacity, *state_shape), dtype=np.uint8)
        self["action"] = np.zeros(
            (self.capacity, *action_shape), dtype=np.float32)
        self["reward"] = np.zeros(
            (self.capacity, ), dtype=np.float32)
        self["next_state"] = np.zeros(
            (self.capacity, *state_shape), dtype=np.uint8)
        self["done"] = np.zeros(
            (self.capacity, ), dtype=np.float32)
        self["priority"] = np.zeros(
            (self.capacity, ), dtype=np.float32)

    def __len__(self):
        return self._n

    def push(self, state, action, reward, next_state, done, error):
        self["state"][self._p] = (state*255).astype(np.uint8)
        self["action"][self._p] = action
        self["reward"][self._p] = reward
        self["next_state"][self._p] = (next_state*255).astype(np.uint8)
        self["done"][self._p] = done
        self["priority"][self._p] = self._priority(error)

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def update_priority(self, idx, error):
        self["priority"][idx] = self._priority(error)

    def _priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        # anneal beta
        self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
        # sample data
        sampler = WeightedRandomSampler(
            self["priority"][:self._n], batch_size, replacement=False)
        idx = list(sampler)

        # create batch
        batch = dict()
        for key in self.keys:
            batch[key] = self[key][idx]

        batch['state'] = torch.FloatTensor(
            batch['state'].astype(np.float32)/255.).to(self.device)
        batch['next_state'] = torch.FloatTensor(
            batch['next_state'].astype(np.float32)/255.).to(self.device)
        batch['action'] = torch.FloatTensor(
            batch['action']).to(self.device)
        batch['reward'] = torch.FloatTensor(
            batch['reward']).unsqueeze(1).to(self.device)
        batch['done'] = torch.FloatTensor(
            batch['done']).unsqueeze(1).to(self.device)

        # IS weights
        p = self["priority"][idx] / np.sum(self["priority"][:self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)

        return batch, idx, weights
