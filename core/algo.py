import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import DiscretePolicy, QNetwork


class SacDiscrete(object):

    def __init__(self, observation_space, action_space, configs):
        self.device = torch.device(
            "cuda" if configs.cuda and torch.cuda.is_available() else "cpu")
        self.gamma = configs.gamma
        self.lr = configs.lr
        self.grad_clip = configs.grad_clip
        self.multi_step = configs.multi_step
        self.update_type = configs.update_type
        self.tau = configs.tau
        self.target_updates_per_steps = configs.target_updates_per_steps
        self.start_steps = configs.start_steps
        self.num_steps = configs.num_steps
        self.use_per = configs.per

        # ---- critic ---- #
        # network
        self.critic = QNetwork(
            observation_space.shape[0], action_space.n
            ).to(device=self.device)
        # target network
        self.critic_target = QNetwork(
            observation_space.shape[0], action_space.n
            ).to(device=self.device).eval()
        # optimizer
        self.critic1_optim = Adam(
            self.critic.Q1.parameters(), lr=self.lr, eps=1e-4)
        self.critic2_optim = Adam(
            self.critic.Q2.parameters(), lr=self.lr, eps=1e-4)
        # copy parameters to the target network
        self.hard_update()

        # ---- entropy ---- #
        self.target_entropy = -np.log(1.0/action_space.n) * 0.98
        self.target_annealing = self.target_entropy * 0.2 *\
            configs.update_every_n_steps / self.num_steps
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr, eps=1e-4)

        # ---- actor ---- #
        # network
        self.policy = DiscretePolicy(
            observation_space.shape[0], action_space.n
            ).to(self.device)
        # optimizer
        self.policy_optim = Adam(
            self.policy.parameters(), lr=self.lr, eps=1e-4)

    def _step_target_entropy(self):
        self.target_entropy -= self.target_annealing

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # stochastic
        if eval is False:
            action, _, _ = self.policy.sample(state)
        # deterministic
        else:
            with torch.no_grad():
                action = self.policy.act(state)
        return action.detach().cpu().numpy()[0]

    def calc_critic_loss(self, state_batch, action_batch, reward_batch,
                         next_state_batch, done_batch, weight=None):
        with torch.no_grad():
            # sample action
            next_state_action, action_probs, log_action_probs =\
                self.policy.sample(next_state_batch)
            # next targets: (num_batches, 1)
            qf1_next_target, qf2_next_target =\
                self.critic_target(next_state_batch)
            # (num_batches, num_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = action_probs * (
                min_qf_next_target - self.alpha * log_action_probs)
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
            # next Q values: (num_batches, 1)
            next_q_value = reward_batch +\
                (1.0 - done_batch) * (self.gamma ** self.multi_step)\
                * min_qf_next_target

        # (num_batches, num_actions)
        qf1, qf2 = self.critic(state_batch)
        # (num_batches, 1)
        qf1 = qf1.gather(1, action_batch.long())
        qf2 = qf2.gather(1, action_batch.long())

        # loss
        if weight is None:
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
        else:
            qf1_loss = ((qf1 - next_q_value).pow(2) * weight).mean()
            qf2_loss = ((qf2 - next_q_value).pow(2) * weight).mean()

        return qf1_loss, qf2_loss, qf1.clone().detach().mean(),\
            qf2.clone().detach().mean()

    def calc_actor_loss(self, state_batch, weight=None):
        # sample action and probs
        action, action_probs, log_action_probs =\
            self.policy.sample(state_batch)
        # soft Q function: (num_batches, num_actions)
        qf1_pi, qf2_pi = self.critic(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # loss
        inside_term = self.alpha * log_action_probs - min_qf_pi
        inside_term = torch.sum(
            action_probs * inside_term, dim=1, keepdim=True)
        if weight is None:
            policy_loss = inside_term.mean()
        else:
            policy_loss = (weight * inside_term).mean()
        # entropies: (num_batches, 1)
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)
        return policy_loss, entropies

    def calculate_alpha_loss(self, entropies, weight=None):
        if weight is None:
            alpha_loss = -(
                self.log_alpha *
                (-entropies + self.target_entropy).detach()
                ).mean()
        else:
            alpha_loss = -(
                self.log_alpha * weight *
                (-entropies + self.target_entropy).detach()
                ).mean()
        return alpha_loss

    def update_parameters(self, memory, batch_size, total_steps, writer):
        if self.use_per:
            batch, idx, weight = memory.sample(batch_size=batch_size)
            state_batch = batch["state"]
            action_batch = batch["action"]
            reward_batch = batch["reward"]
            next_state_batch = batch["next_state"]
            done_batch = batch["done"]
            weight = torch.FloatTensor(weight).unsqueeze(-1).to(self.device)
        else:
            state_batch, action_batch, reward_batch, next_state_batch,\
                done_batch = memory.sample(batch_size=batch_size)
            weight = None

        # loss of critics
        qf1_loss, qf2_loss, mean_Q1, mean_Q2 = self.calc_critic_loss(
            state_batch, action_batch, reward_batch, next_state_batch,
            done_batch, weight)

        # loss of the actor
        policy_loss, entropies = self.calc_actor_loss(state_batch, weight)

        # loss of the alpha
        alpha_loss = self.calculate_alpha_loss(entropies, weight)

        if self.use_per:
            error = self.calc_q(
                state_batch, action_batch, reward_batch, next_state_batch,
                done_batch)
            memory.update_priority(idx, error.reshape(-1))

        # update
        self._update(
            self.critic1_optim, self.critic.Q1, qf1_loss,
            grad_clip=self.grad_clip)
        self._update(
            self.critic2_optim, self.critic.Q2, qf2_loss,
            grad_clip=self.grad_clip)
        self._update(
            self.policy_optim, self.policy.policy, policy_loss,
            grad_clip=self.grad_clip)
        self._update(
            self.alpha_optim, None, alpha_loss)

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()

        # soft update
        if self.update_type == 'soft':
            self.soft_update()
        # hard update
        elif total_steps % self.target_updates_per_steps == 0:
            self.hard_update()

        self._step_target_entropy()

        writer.add_scalar(
            'loss/critic_1', qf1_loss.item(), total_steps)
        writer.add_scalar(
            'loss/critic_2', qf2_loss.item(), total_steps)
        writer.add_scalar(
            'loss/policy', policy_loss.item(), total_steps)
        writer.add_scalar(
            'loss/alpha', alpha_loss.item(), total_steps)
        writer.add_scalar(
            'stats/alpha', alpha_tlogs.item(), total_steps)
        writer.add_scalar(
            'stats/mean_Q1', mean_Q1.item(), total_steps)
        writer.add_scalar(
            'stats/mean_Q2', mean_Q2.item(), total_steps)
        writer.add_scalar(
            'stats/mean_entropy',
            entropies.clone().detach().mean().item(),
            total_steps)

    def _update(self, optim, network, loss, grad_clip=None):
        optim.zero_grad()
        loss.backward(retain_graph=False)
        if grad_clip is not None:
            for p in network:
                torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
        optim.step()

    def get_batch(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor([action]).unsqueeze(-1).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(
            next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(-1).to(self.device)
        return state, action, reward, next_state, done

    def calc_q(self, state, action, reward, next_state, done):
        with torch.no_grad():
            curr_q1, _ = self.critic(state)
            curr_q1 = curr_q1.gather(1, action.long())
            next_action, action_prob, log_action_prob =\
                self.policy.sample(next_state)
            next_q1, next_q2 =\
                self.critic_target(next_state)
            next_q_min = (action_prob * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_prob
                )).mean(dim=1, keepdim=True)
            next_q_value = reward + (1.0 - done)\
                * (self.gamma ** self.multi_step) * next_q_min

        return torch.abs(curr_q1 - next_q_value).cpu().numpy()

    def soft_update(self):
        for target, source in zip(
                self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(
                target.data * (1.0 - self.tau) + source.data * self.tau)

    def hard_update(self):
        for target, source in zip(
                self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(source.data)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.policy.save(os.path.join(save_dir, 'actor.pth'))
        self.critic.save(os.path.join(save_dir, 'critic.pth'))

    def load_model(self, save_dir):
        self.policy.load(os.path.join(save_dir, 'actor.pth'))
        self.critic.load(os.path.join(save_dir, 'critic.pth'))
