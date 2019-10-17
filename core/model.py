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
            # (84, 84, input_channels) -> (20, 20, 32)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # (20, 20, 32) -> (9, 9, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # (9, 9, 64) -> (7, 7, 64)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            Flatten(),
            # (7 * 7 * 64, ) -> (512, )
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
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2


class DiscretePolicy(BaseNetwork):
    """ Discrete policy. """

    def __init__(self, input_channels, num_actions):
        super(DiscretePolicy, self).__init__()

        # policy network
        self.policy = nn.Sequential(
            # (84, 84, input_channels) -> (20, 20, 32)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # (20, 20, 32) -> (9, 9, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # (9, 9, 64) -> (7, 7, 64)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            Flatten(),
            # (7 * 7 * 64, ) -> (512, )
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            # (512, ) -> (num_actions, )
            nn.Linear(512, num_actions),
            nn.Softmax(dim=1)
        ).apply(weights_init_he)

    def sample(self, state):
        # action probabilities
        action_probs = self.policy(state)
        # actions with maximum probabilities
        max_prob_actions = torch.argmax(action_probs).unsqueeze(0)
        # distribution of policy's actions
        action_dist = Categorical(action_probs)
        # sample actions
        actions = action_dist.sample().cpu()
        # log likelihood of actions
        log_action_probs = torch.log(action_probs + 1e-8)

        return actions, action_probs, log_action_probs, max_prob_actions
