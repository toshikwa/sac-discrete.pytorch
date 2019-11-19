import os
import argparse
from datetime import datetime
from rltorch.env import make_pytorch_env

from agent import SacDiscreteAgent


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': 300000,
        'batch_size': 64,
        'target_entropy_ratio': 0.95,  # 0.98 in the paper
        'lr': 0.0003,
        'memory_size': 1e6,
        'gamma': 0.99,
        'target_update_type': 'soft',  # 'soft' or 'hard'
        'tau': 0.005,  # ignored if update_type='hard'
        'target_update_interval': 8000,  # ignored if update_type='soft'
        'multi_step': 3,
        'per': True,  # prioritized experience replay
        'alpha': 0.6,  # ignored when per=False
        'beta': 0.4,  # ignored when per=False
        'beta_annealing': 0.0001,  # ignored when per=False
        'grad_clip': 5.0,
        'update_every_n_steps': 4,
        'learnings_per_update': 1,
        'start_steps': 20000,
        'log_interval': 10,
        'eval_interval': 5000,
        'cuda': args.cuda,
        'seed': args.seed
    }

    env = make_pytorch_env(args.env_id, clip_rewards=False)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id,
        f'sac-discrete-seed{args.seed}-{time}')

    agent = SacDiscreteAgent(
        env=env, test_env=test_env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
