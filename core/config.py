import argparse


def get_configs():
    parser = argparse.ArgumentParser()
    # SAC-Discrete configs
    parser.add_argument(
        '--env_name', type=str, default='MsPacmanNoFrameskip-v4',
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
        '--update_type', choices=['soft', 'hard'], default='soft',
        help='Type of target update.')
    parser.add_argument(
        '--tau', type=float, default=0.005,
        help='Target smoothing coefficient for soft update.')
    parser.add_argument(
        '--target_updates_per_steps', type=int, default=8000,
        help='Target updates per iterations for hard update.')
    parser.add_argument(
        '--grad_clip', type=float, default=5.0,
        help='Gradient norm clipping.')
    parser.add_argument(
        '--multi_step', type=int, default=3,
        help='Multi-step rewards.')
    parser.add_argument(
        '--update_every_n_steps', type=int, default=4,
        help='Environment steps per update.')
    parser.add_argument(
        '--learning_per_update', type=int, default=1,
        help='Learning updates per learning.')
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
        '--eval_per_iters', type=int, default=5000,
        help='Evaluation per iterations.')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed.')
    parser.add_argument(
        '--num_steps', type=int, default=100000,
        help='Maximum number of steps.')
    parser.add_argument(
        '--cuda', action="store_true",
        help='If use gpu or not.')

    return parser.parse_args()
