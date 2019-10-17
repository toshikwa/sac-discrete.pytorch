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
        self.target_updates_per_iters = configs.target_updates_per_iters
        self.start_steps = configs.start_steps

        # ---- critic ---- #
        # network
        self.critic = QNetwork(
            observation_space.shape[2], action_space.n
            ).to(device=self.device)
        # target network
        self.critic_target = QNetwork(
            observation_space.shape[2], action_space.n
            ).to(device=self.device).eval()
        # optimizer
        self.critic_optim = Adam(
            self.critic.parameters(), lr=self.lr, eps=1e-4)
        # copy parameters to the target network
        self.hard_update()

        # ---- entropy ---- #
        self.target_log_entropy = -np.log((1.0 / action_space.n)) * 0.98
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=configs.lr, eps=1e-4)

        # ---- actor ---- #
        # network
        self.policy = DiscretePolicy(
            observation_space.shape[2], action_space.n
            ).to(self.device)
        # optimizer
        self.policy_optim = Adam(
            self.policy.parameters(), lr=configs.lr, eps=1e-4)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # stochastic
        if eval is False:
            action, _, _, _ = self.policy.sample(state)
        # deterministic
        else:
            _, _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def calc_critic_loss(self, state_batch, action_batch, reward_batch,
                         next_state_batch, mask_batch):

        with torch.no_grad():
            # sample action
            next_state_action, action_probs, log_action_probs, _ =\
                self.policy.sample(next_state_batch)
            # next targets: (num_batches, 1)
            qf1_next_target, qf2_next_target =\
                self.critic_target(next_state_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = (action_probs * (
                min_qf_next_target - self.alpha * log_action_probs
                )).mean(dim=1).unsqueeze(-1)
            # next Q values: (num_batches, 1)
            next_q_value = reward_batch +\
                mask_batch * self.gamma * (min_qf_next_target)

        # Q_i: (num_batches, 1)
        qf1, qf2 = self.critic(state_batch)
        qf1 = qf1.gather(1, action_batch.long())
        qf2 = qf2.gather(1, action_batch.long())
        # loss
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        return qf1_loss, qf2_loss

    def calc_actor_loss(self, state_batch):
        # sample action and probs
        action, action_probs, log_action_probs, _ =\
            self.policy.sample(state_batch)
        # soft Q function: (num_batches, num_actions)
        qf1_pi, qf2_pi = self.critic(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # loss
        policy_loss = (
            (self.alpha * log_action_probs - min_qf_pi) * action_probs
            ).mean()
        # negative entropies: (num_batches, )
        negative_entropies = torch.sum(
            log_action_probs * action_probs, dim=1)
        return policy_loss, negative_entropies

    def calculate_alpha_loss(self, negative_entropies):
        alpha_loss = -(
            self.log_alpha *
            (negative_entropies + self.target_log_entropy).detach()
            ).mean()
        return alpha_loss

    def update_parameters(self, memory, batch_size, updates):
        # sample batches
        state_batch, action_batch, reward_batch,\
            next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # current state: (num_batches, 4, 84, 84)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next state: (num_batches, 4, 84, 84)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action: (num_batches, 1)
        action_batch = torch.FloatTensor(
            action_batch).to(self.device).unsqueeze(1)
        # reward: (num_batches, 1)
        reward_batch = torch.FloatTensor(
            reward_batch).to(self.device).unsqueeze(1)
        # mask: (num_batches, 1)
        mask_batch = torch.FloatTensor(
            mask_batch).to(self.device).unsqueeze(1)

        # loss of critics
        qf1_loss, qf2_loss = self.calc_critic_loss(
            state_batch, action_batch, reward_batch, next_state_batch,
            mask_batch)

        # loss of the actor
        policy_loss, negative_entropies = self.calc_actor_loss(state_batch)

        # loss of the alpha
        alpha_loss = self.calculate_alpha_loss(negative_entropies)

        # update Q1
        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optim.step()
        # update Q2
        self.critic_optim.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.critic_optim.step()
        # update pi
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optim.step()
        # update alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()

        if updates % self.target_updates_per_iters == 0:
            self.hard_update()

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(),\
            alpha_loss.item(), alpha_tlogs.item()

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
