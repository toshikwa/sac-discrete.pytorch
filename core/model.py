import torch
import torch.nn as nn
from torch.distributions import Categorical


def weights_init_he(m):
    """ Initialize weights with He's initializer. """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def grad_false(m):
    for param in m.parameters():
        param.requires_grad = False


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def eval(self):
        return super(BaseNetwork, self).eval().apply(grad_false)


class QNetwork(BaseNetwork):
    """ Pairs of two Q-networks. """

    def __init__(self, input_channels, num_actions):
        super(QNetwork, self).__init__()

        # Q1
        self.Q1 = nn.Sequential(
            # (input_channels, 84, 84) -> (32, 20, 20)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # (32, 20, 20) -> (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # (64, 9, 9) -> (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            Flatten(),
            # (64 * 7 * 7, ) -> (512, )
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            # (512, ) -> (num_actions, )
            nn.Linear(512, num_actions),
        ).apply(weights_init_he)

        # Q2 (exactly the same structure with Q1)
        self.Q2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        ).apply(weights_init_he)

    def forward(self, state):
        # Q_i: (num_batches, num_actions)
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2


class DiscretePolicy(BaseNetwork):
    """ Discrete policy. """

    def __init__(self, input_channels, num_actions):
        super(DiscretePolicy, self).__init__()

        # policy network (exactly the same structure with critics)
        self.policy = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
            nn.LogSoftmax(dim=1)
        ).apply(weights_init_he)

    def sample(self, state):
        # log action probabilities: (num_batches, num_actions)
        log_action_probs = self.policy(state)
        # action probabilities: (num_batches, num_actions)
        action_probs = log_action_probs.exp()
        # greedy actions: (num_batches, )
        max_prob_actions = torch.argmax(action_probs, dim=1)
        # distribution of policy's actions
        action_dist = Categorical(action_probs)
        # stochastic actions: (num_batches, )
        actions = action_dist.sample().cpu()

        return actions, action_probs, log_action_probs, max_prob_actions
