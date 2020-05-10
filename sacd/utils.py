from collections import deque
import numpy as np
import torch


def update_params(optim, loss, network, retain_graph=False,
                  grad_cliping=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_cliping)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
