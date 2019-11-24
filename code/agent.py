import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from memory import DummyMultiStepMemory, DummyPrioritizedMemory
from model import TwinnedQNetwork, CateoricalPolicy
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats


class SacDiscreteAgent:

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 target_entropy_ratio=0.98, lr=0.0003, memory_size=1000000,
                 gamma=0.99, target_update_type='soft',
                 target_update_interval=8000, tau=0.005, multi_step=1,
                 per=False, alpha=0.6, beta=0.4, beta_annealing=0.0001,
                 grad_clip=5.0, update_every_n_steps=4,
                 learnings_per_update=1, start_steps=1000, log_interval=10,
                 eval_interval=1000, cuda=True, seed=0):
        self.env = env
        self.test_env = test_env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[0], self.env.action_space.n
            ).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n
            ).to(device=self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n
            ).to(device=self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=1e-4)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr, eps=1e-4)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr, eps=1e-4)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy =\
            -np.log(1.0/self.env.action_space.n) * target_entropy_ratio
        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr, eps=1e-4)

        # DummyMemory efficiently stores FrameStacked states.
        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = DummyPrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = DummyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.target_update_type = target_update_type
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.per = per
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.grad_clip = grad_clip
        self.update_every_n_steps = update_every_n_steps
        self.learnings_per_update = learnings_per_update
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_every_n_steps == 0\
            and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # act without randomness
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs =\
                self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).mean(dim=1, keepdim=True)

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            # clip reward to [-1.0, 1.0]
            clipped_reward = max(min(reward, 1.0), -1.0)

            if self.per:
                batch = to_batch(
                    state, action, clipped_reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    curr_q1, _ = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, clipped_reward, next_state, masked_done,
                    error, episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, clipped_reward, next_state, masked_done,
                    episode_done=done)

            if self.is_update():
                for _ in range(self.learnings_per_update):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()

            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1
        if self.target_update_type == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.learning_steps % self.target_update_interval == 0:
            hard_update(self.critic_target, self.critic)

        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)
        update_params(self.alpha_optim, None, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (log of) probabilities to calculate expectations of Q and entropies
        _, action_probs, log_action_probs = self.policy.sample(states)
        # Q for every actions to calculate expectations of Q
        q1, q2 = self.critic(states)
        q = torch.min(q1, q2)

        # expectations of entropies
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)
        # expectations of Q
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.test_env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_return:<5.1f} +/- {std_return:<5.1f}')
        print('-' * 60)

    def save_models(self):
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
