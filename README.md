# Soft Actor-Critic in PyTorch
A PyTorch implementation of SAC-Discrete[[1]](#references) with n-step rewards and prioritized experience replay[[2]](#references). It's based on [the auther's implementation](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch), however it doesn't support for atari environments.

If you want to train a distributed version of SAC-Discrete or a continuous version of Soft Actor-Critic, please reffer to [rltorch](https://github.com/ku2482/rltorch) repository or [Soft Actor-Critic](https://github.com/ku2482/soft-actor-critic.pytorch) repository respectively.

## Requirements
You can install liblaries using `pip install -r requirements.txt`.

## Examples
You can train SAC-Discrete agent like this example [here](https://github.com/ku2482/sac-discrete.pytorch/blob/master/code/main.py).

```
python code/main.py \
[--env_id str(default MsPacmanNoFrameskip-v4)] \
[--cuda (optional)] \
[--seed int(default 0)]
```

If you want to use n-step rewards and prioritized experience replay, set `multi_step=3` and `per=True` in configs.

## Results

I just teseted in **MsPacman** environment. A result after 100,000 steps will be around 600, which is comparable with results of the paper.

![mspacman](https://user-images.githubusercontent.com/37267851/67738428-e4d57480-fa51-11e9-94b2-1492760e3907.gif)

## References
[[1]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).

[[2]](https://arxiv.org/abs/1511.05952) Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).