# SAC-Discrete in PyTorch
This is a PyTorch implementation of SAC-Discrete[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Installation
If you are using Anaconda, first create the virtual environment.

```bash
conda create -n sacd python=3.7 -y
conda activate sacd
```

You can install Python liblaries using pip.

```bash
pip install -r requirements.txt
```

## Examples
You can train SAC-Discrete agent like this example [here](https://github.com/ku2482/sac-discrete.pytorch/blob/master/code/main.py).

```
python train.py --config config/sacd.yaml --env_id MsPacmanNoFrameskip-v4 --cuda --seed 0
```

## Results
Results of above example (with n-step rewards and prioritized experience replay) will be like below, which is comparable (if no better) with the paper.
Note that scores reported in the paper are evaluated at 1e5 steps.

<img src="https://user-images.githubusercontent.com/37267851/69165567-23cc8680-0b35-11ea-8a3c-b251bacce975.png" title="graph" width=500><img src="https://user-images.githubusercontent.com/37267851/67809830-c9fc1200-fadc-11e9-8f48-799a19689dd6.gif" title="gif" width=300>

## References
[[1]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).
