# SAC-Discrete in PyTorch
This is a PyTorch implementation of SAC-Discrete[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

**UPDATE**
- 2020.5.10
    - Refactor codes and fix a bug of SAC-Discrete algorithm.
    - Implement Prioritized Experience Replay[[2]](#references), N-step Return and Dueling Networks[[3]](#references).
    - Test them.

## Setup
If you are using Anaconda, first create the virtual environment.

```bash
conda create -n sacd python=3.7 -y
conda activate sacd
```

You can install Python liblaries using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.


## Examples
You can train SAC-Discrete agent like this example [here](https://github.com/ku2482/sac-discrete.pytorch/blob/master/train.py).

```
python train.py --config config/sacd.yaml --env_id MsPacmanNoFrameskip-v4 --cuda --seed 0
```

If you want to use Prioritized Experience Replay(PER), N-step return or Dueling Networks, change `use_per`, `multi_step` or `dueling_net` respectively.

## Results
I just evaluated vanilla SAC-Discrite, with PER, N-step Return or Dueling Networks in `MsPacmanNoFrameskip-v4`. The graph below shows the test returns along with environment steps (which equals environment frames divided by the factor of 4). Also, note that curves are smoothed by exponential moving average with `weight=0.5` for visualization.

<img src="https://user-images.githubusercontent.com/37267851/81498474-319edf80-9300-11ea-9353-a9055062eef5.png" title="graph" width=500><img src="https://user-images.githubusercontent.com/37267851/67809830-c9fc1200-fadc-11e9-8f48-799a19689dd6.gif" title="gif" width=300>

N-step Return and PER seems helpful to better utilize RL signals (e.g. sparse rewards).

## References
[[1]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).

[[2]](https://arxiv.org/abs/1511.05952) Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

[[3]](https://arxiv.org/abs/1511.06581) Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." arXiv preprint arXiv:1511.06581 (2015).
