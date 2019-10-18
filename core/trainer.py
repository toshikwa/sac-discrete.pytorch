import os
import datetime
import itertools
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import make_atari_game
from algo import SacDiscrete
from memory import ReplayMemory
from vis import plot_return_history


CORE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(CORE_DIR)


class Trainer():

    def __init__(self, configs):
        # environment
        self.env = make_atari_game(configs.env_name)

        # seed
        torch.manual_seed(configs.seed)
        np.random.seed(configs.seed)
        self.env.seed(configs.seed)

        # agent
        self.agent = SacDiscrete(
            self.env.observation_space, self.env.action_space, configs)

        # logdir
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.logdir = os.path.join(
            HOME_DIR, 'logs', configs.env_name, now)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.logdir, 'summary'))
        # replay memory
        self.memory = ReplayMemory(configs.replay_buffer_size)

        # return history
        self.mean_return_history = np.array([], dtype=np.float)
        self.std_return_history = np.array([], dtype=np.float)

        self.env_name = configs.env_name
        self.vis = configs.vis
        self.batch_size = configs.batch_size
        self.start_steps = configs.start_steps
        self.update_every_n_steps = configs.update_every_n_steps
        self.learning_per_update = configs.learning_per_update
        self.eval_per_iters = configs.eval_per_iters
        self.num_steps = configs.num_steps
        self.max_episode_steps = self.env.spec.tags.get(
            'wrapper_config.TimeLimit.max_episode_steps')

        # training steps
        self.total_numsteps = 0
        # update counts
        self.updates = 0

    def update(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.learning_per_update):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha =\
                    self.agent.update_parameters(
                        self.memory, self.batch_size, self.updates)

                self.writer.add_scalar(
                    'loss/critic_1', critic_1_loss, self.updates)
                self.writer.add_scalar(
                    'loss/critic_2', critic_2_loss, self.updates)
                self.writer.add_scalar(
                    'loss/policy', policy_loss, self.updates)
                self.writer.add_scalar(
                    'loss/entropy_loss', ent_loss, self.updates)
                self.writer.add_scalar(
                    'entropy_temprature/alpha', alpha, self.updates)
                self.updates += 1

    def train_episode(self, episode):
        # rewards
        episode_reward = 0
        # steps
        episode_steps = 0
        # done
        done = False
        # initial state
        state = self.env.reset()

        while not done:
            if self.vis:
                self.env.render()

            for _ in range(self.update_every_n_steps):
                # take the random action
                if self.start_steps > self.total_numsteps:
                    action = self.env.action_space.sample()
                # sample from the policy
                else:
                    action = self.agent.select_action(state)
                # act
                next_state, reward, done, _ = self.env.step(action)
                episode_steps += 1
                episode_reward += reward
                self.total_numsteps += 1

                # ignore the "done" if it comes from hitting the time horizon.
                mask = 1 if episode_steps == self.max_episode_steps\
                    else float(not done)

                # store in the replay memory
                self.memory.push(state, action, reward, next_state, mask)
                state = next_state

            self.update()

            if self.total_numsteps % self.eval_per_iters == 0:
                self.evaluate()

        self.writer.add_scalar('reward/train', episode_reward, episode)
        print(f"Episode: {episode}, "
              f"total numsteps: {self.total_numsteps}, "
              f"episode steps: {episode_steps}, "
              f"total updates: {self.updates}, "
              f"reward: {round(episode_reward, 2)}")

    def evaluate(self):
        # evaluate
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float)
        env = make_atari_game(self.env_name)

        for i in range(episodes):
            state = env.reset()
            episode_reward = 0.
            done = False
            while not done:
                if self.vis:
                    env.render()
                action = self.agent.select_action(state, eval=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        # mean return
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'avg_reward/test', mean_return, self.total_numsteps)

        print("----------------------------------------")
        print(f"Num steps: {self.total_numsteps}, "
              f"Test Return: {round(mean_return, 2)}"
              f" +/- {round(std_return, 2)}")
        print("----------------------------------------")

        # save return
        self.mean_return_history = np.append(
            self.mean_return_history, mean_return)
        self.std_return_history = np.append(
            self.std_return_history, std_return)
        np.save(
            os.path.join(self.logdir, 'mean_return_history.npy'),
            self.mean_return_history)
        np.save(
            os.path.join(self.logdir, 'std_return_history.npy'),
            self.std_return_history)

        # plot
        plot_return_history(
            self.mean_return_history, self.std_return_history,
            os.path.join(self.logdir, 'test_rewards.png'),
            self.env_name, self.eval_per_iters)

    def train(self):
        # iterate until convergence
        for episode in itertools.count(1):
            # train
            self.train_episode(episode)
            if self.updates > self.num_steps:
                break

        self.agent.save_model(
            os.path.join(self.logdir, "model"))

    def __del__(self):
        self.env.close()
