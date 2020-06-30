import numpy as np
import torch

from .base import MultiStepBuff


class FlatMemory(dict):

    def __init__(self, capacity, state_shape, device):
        super(FlatMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        self['state'] = np.empty(
            (self.capacity, *self.state_shape), dtype=np.float32)
        self['next_state'] = np.empty(
            (self.capacity, *self.state_shape), dtype=np.float32)

        self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'][self._p] = state
        self['next_state'][self._p] = next_state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        states = torch.FloatTensor(
            self['state'][indices]).to(self.device)
        next_states = torch.FloatTensor(
            self['next_state'][indices]).to(self.device)
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n


class FlatMultiStepMemory(FlatMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(FlatMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)
