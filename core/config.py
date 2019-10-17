import argparse


def get_configs():
    parser = argparse.ArgumentParser()
    # SAC-Discrete configs
    parser.add_argument(
        '--env_name', type=str, default='MsPacman-v0',
        help='Name of the environment to run.')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size.')
    parser.add_argument(
        '--replay_buffer_size', type=int, default=1000000,
        help='Buffer size.')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor for the reward.')
    parser.add_argument(
        '--env_steps_per_iters', type=int, default=4,
        help='Environment steps per iterations.')
    parser.add_argument(
        '--updates_per_iters', type=int, default=1,
        help='Learning updates per iterations.')
    parser.add_argument(
        '--target_updates_per_iters', type=int, default=8000,
        help='Target updates per iterations.')
    parser.add_argument(
        '--lr', type=float, default=0.0003,
        help='Learning rate.')
    parser.add_argument(
        '--start_steps', type=int, default=20000,
        help='Steps sampling random actions.')

    # training configs
    parser.add_argument(
        '-v', '--vis', action="store_true",
        help='If render the environment or not.')
    parser.add_argument(
        '--eval_per_iters', type=int, default=10000,
        help='Evaluation per iterations.')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed.')
    parser.add_argument(
        '--num_steps', type=int, default=100000,
        help='Maximum number of steps.')
    parser.add_argument(
        '--cuda', action="store_true",
        help='If use gpu or not.')

    return parser.parse_args()
