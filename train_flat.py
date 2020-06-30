import os
import yaml
import argparse
from datetime import datetime
import gym

from sacd.agent import FlatSacdAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = gym.make(args.env_id)
    test_env = gym.make(args.env_id)

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id,
        f'sacd-tent{config["target_entropy_ratio"]}-{time}')

    # Create the agent.
    agent = FlatSacdAgent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/flat.yaml')
    parser.add_argument('--env_id', type=str, default='CartPole-v0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
