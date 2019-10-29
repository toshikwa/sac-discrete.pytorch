import os
import datetime
import itertools
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gym import wrappers

from env import make_pytorch_env
from algo import SacDiscrete
from memory import ReplayMemory, MultiStepBuff
from per import PER


CORE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(CORE_DIR)


class Trainer():

    def __init__(self, configs):
        self.device = torch.device(
            "cuda" if configs.cuda and torch.cuda.is_available() else "cpu")
        # environment
        self.env = make_pytorch_env(configs.env_name)

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

        # memory
        self.buff = MultiStepBuff(capacity=self.agent.multi_step)
        if self.agent.use_per:
            self.memory = PER(
                configs.replay_buffer_size, self.env.observation_space.shape,
                (1,), self.device)
        else:
            self.memory = ReplayMemory(
                configs.replay_buffer_size, self.device)

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

        self.total_steps = 0
        self.updates = 0

    def update(self):
        self.agent.update_parameters(
            self.memory, self.batch_size, self.total_steps, self.writer)
        self.updates += 1

    def train_episode(self, episode):
        # rewards
        episode_reward = 0.
        # steps
        episode_steps = 0
        # done
        done = False
        # initial state
        state = self.env.reset()
        self.buff.reset()

        while not done:
            if self.vis:
                self.env.render()

            # take the random action
            if self.start_steps > self.total_steps:
                action = self.env.action_space.sample()
            # sample from the policy
            else:
                action = self.agent.select_action(state)
            # act
            next_state, reward, done, _ = self.env.step(action)
            episode_steps += 1
            episode_reward += reward
            self.total_steps += 1

            # ignore the "done" if it comes from hitting the time horizon
            masked_done = False if episode_steps >= self.max_episode_steps\
                else done

            # clip reward
            clipped_reward = max(min(reward, 1.0), -1.0)

            self.buff.push(
                state, action, clipped_reward, next_state, masked_done)

            # store in the replay memory
            if len(self.buff) == self.agent.multi_step:
                self.append_memory()

            if done:
                while len(self.buff) > 0:
                    self.append_memory()
                    self.buff.memory["reward"].append(0.0)

            state = next_state

            if self._is_eval():
                self.evaluate()

            if self._is_update():
                for _ in range(self.learning_per_update):
                    self.update()

        self.writer.add_scalar('reward/train', episode_reward, episode)
        print(f"Episode: {episode}, "
              f"total steps: {self.total_steps}, "
              f"episode steps: {episode_steps}, "
              f"total updates: {self.updates}, "
              f"reward: {round(episode_reward, 2)}")

    def append_memory(self):
        state, action, reward, next_state, done =\
            self.buff.get(self.agent.gamma)
        if self.agent.use_per:
            error = self.agent.calc_q(*self.agent.get_batch(
                state, action, reward, next_state, done)).item()
            self.memory.push(
                state, action, reward, next_state, done, error)
        else:
            self.memory.push(
                state, action, reward, next_state, done)

    def _is_eval(self):
        return self.total_steps % self.eval_per_iters == 0

    def _is_update(self):
        return len(self.memory) > self.batch_size and\
            self.total_steps % self.update_every_n_steps == 0

    def evaluate(self):
        # evaluate
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float)
        env = make_pytorch_env(self.env_name)
        env = wrappers.Monitor(
            env, os.path.join(
                self.logdir, 'monitor', f'step-{self.total_steps}'),
            video_callable=lambda x: True)
        action_bar = np.zeros((env.action_space.n), np.int)

        for i in range(episodes):
            state = env.reset()
            episode_reward = 0.
            done = False
            while not done:
                if self.vis:
                    env.render()
                action = self.agent.select_action(state, eval=True)
                action_bar[action] += 1
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        # mean return
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.total_steps)

        print("----------------------------------------")
        print(f"Num steps: {self.total_steps}, "
              f"Test Return: {round(mean_return, 2)}"
              f" +/- {round(std_return, 2)}")
        print("Actions: ", np.round(action_bar / action_bar.sum(), 3))
        print("----------------------------------------")

    def train(self):
        # iterate until convergence
        for episode in itertools.count(1):
            # train
            self.train_episode(episode)
            if self.total_steps > self.num_steps:
                break

        self.agent.save_model(
            os.path.join(self.logdir, "model"))

    def __del__(self):
        self.env.close()
