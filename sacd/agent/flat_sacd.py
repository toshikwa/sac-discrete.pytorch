import torch
from torch.optim import Adam

from sacd.agent import SacdAgent
from sacd.utils import disable_gradients
from sacd.model import FlatTwinnedQNetwork, FlatCateoricalPolicy


class FlatSacdAgent(SacdAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0, units=[256, 256]):
        super(FlatSacdAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, lr, memory_size,
            gamma, multi_step, target_entropy_ratio, start_steps,
            update_interval, target_update_interval, False, dueling_net,
            num_eval_steps, max_episode_steps, log_interval, eval_interval,
            cuda, seed)

        del self.policy
        del self.online_critic
        del self.target_critic
        del self.policy_optim
        del self.q1_optim
        del self.q2_optim

        # Define networks.
        self.policy = FlatCateoricalPolicy(
            self.env.observation_space.shape[0], self.env.action_space.n, units
            ).to(self.device)
        self.online_critic = FlatTwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            units, dueling_net=dueling_net).to(self.device)
        self.target_critic = FlatTwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            units, dueling_net=dueling_net).to(self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

    def explore(self, state):
        # Act with randomness.
        state = torch.FloatTensor(state[None, ...]).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.FloatTensor(state[None, ...]).to(self.device)
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()
