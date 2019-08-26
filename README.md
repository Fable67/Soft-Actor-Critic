**This repository is still a work in progress. Pull requests are happily welcome!**



# Soft Actor-Critic

Soft actor-critic is a deep reinforcement learning framework for training maximum entropy policies in continuous domains. The algorithm is based on the paper [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) presented at ICML 2018.

This implementation uses Pytorch.

#### What is novel about this implementation?

It supports the Combined experience replay introduced by this paper: [A Deeper Look at Experience Replay](https://arxiv.org/abs/1712.01275) <br>
It supports the new Ranger optimizer introduced by this blog post: [New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + LookAhead for the best of both.](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d)


# Getting Started

Soft Actor-Critic can be run locally.

Examples:

Train agent on the Humanoid-v2 mujoco environment and save checkpoints and tensorboard summary to directory Humanoid-v2/
`python3 main.py --env_name=Humanoid-v2 --log_dir=Humanoid-v2`

Continue training the aformentioned agent
`python3 main.py --env_name=Humanoid-v2 --log_dir=Humanoid-v2 --continue_training`

Test the agent trained on Ant-v3 in the environment with weights loaded from Ant-v3/
`python3 main.py --env_name=Ant-v3 --log_dir=Ant-v3 --test --render_testing --num_test_games=10`


## Prerequisites

Most of the models require a [Mujoco](https://www.roboti.us/license.html) license.


# Credits

The soft actor-critic algorithm was developed by Tuomas Haarnoja under the supervision of Prof. [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) and Prof. [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) at UC Berkeley. Special thanks to [Vitchyr Pong](https://github.com/vitchyr), who wrote some parts of the code, and [Kristian Hartikainen](https://github.com/hartikainen) who helped testing, documenting, and polishing the code and streamlining the installation process. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).


# Reference

[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290).  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. ICML, 2018.

[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905).  
Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine. arXiv preprint, 2018.

[Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610v1). <br>
Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba

[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265v1). <br>
Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han

[A Deeper Look at Experience Replay](https://arxiv.org/abs/1712.01275). <br>
Shangtong Zhang, Richard S. Sutton
