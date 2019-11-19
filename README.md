# SAC-Discrete in PyTorch
A PyTorch implementation of SAC-Discrete[[1]](#references) with n-step rewards and prioritized experience replay[[2]](#references). It's based on [the auther's implementation](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch), however it doesn't support for atari environments.

I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions. If you want to train a distributed version of SAC-Discrete or a continuous version of Soft Actor-Critic, please reffer to [rltorch](https://github.com/ku2482/rltorch) repository or [Soft Actor-Critic](https://github.com/ku2482/soft-actor-critic.pytorch) repository respectively.

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

If you want to use n-step rewards and prioritized experience replay, set `multi_step=3` and `per=True` in configs. Also, I modified `target_entropy_ratio` from 0.98 to 0.95, because 0.98 * maximum entropy seems too large for target entropy.

## Results
Results of above example (with n-step rewards and prioritized experience replay) will be like below, which is comparable (if no better) with the paper.
Note that scores reported in the paper are evaluated at 1e5 steps.

<img src="https://user-images.githubusercontent.com/37267851/69165567-23cc8680-0b35-11ea-8a3c-b251bacce975.png" title="graph" width=500><img src="https://user-images.githubusercontent.com/37267851/67809830-c9fc1200-fadc-11e9-8f48-799a19689dd6.gif" title="gif" width=300>

## References
[[1]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).

[[2]](https://arxiv.org/abs/1511.05952) Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
