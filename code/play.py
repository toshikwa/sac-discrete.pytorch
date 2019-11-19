import os
import argparse
import torch
from rltorch.env import make_pytorch_env, wrap_monitor

from model import CateoricalPolicy
from utils import grad_false


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--log_name', type=str, default='sac-discrete-*')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    log_dir = os.path.join('logs', args.env_id, args.log_name)
    if not os.path.exists(os.path.join(log_dir, 'monitor')):
        os.makedirs(os.path.join(log_dir, 'monitor'))

    env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)
    env = wrap_monitor(env, os.path.join(log_dir, 'monitor'))

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    policy = CateoricalPolicy(
        env.observation_space.shape[0], env.action_space.n
        ).to(device).eval()

    policy.load(os.path.join(log_dir, 'model', 'policy.pth'))
    grad_false(policy)

    def exploit(state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.act(state)
        return action.item()

    state = env.reset()
    episode_reward = 0.
    done = False
    while not done:
        if args.render:
            env.render()
        action = exploit(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state


if __name__ == '__main__':
    run()
