import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def weights_init_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def weights_init_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def create_conv(num_channels, num_actions):
    return nn.Sequential(
        # (num_channels, 84, 84) -> (32, 20, 20)
        nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
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
    ).apply(weights_init_he)  # The author of the paper used He's initializer.


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions):
        super(TwinnedQNetwork, self).__init__()
        self.Q1 = create_conv(num_channels, num_actions)
        self.Q2 = create_conv(num_channels, num_actions)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions):
        super(CateoricalPolicy, self).__init__()
        self.policy = create_conv(num_channels, num_actions)

    def act(self, state):
        # act with greedy policy
        action_logits = self.policy(state)
        greedy_actions = torch.argmax(action_logits, dim=1)
        return greedy_actions

    def sample(self, state):
        # act with exploratory policy
        action_probs = F.softmax(self.policy(state), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # avoid numerical instability
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
