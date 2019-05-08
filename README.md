**This repository is still a work in progress. Pull requests are happily welcome!**



# Soft Actor-Critic

Soft actor-critic is a deep reinforcement learning framework for training maximum entropy policies in continuous domains. The algorithm is based on the paper [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) presented at ICML 2018.

This implementation uses Pytorch.


# Getting Started

Soft Actor-Critic can be run locally.

## Prerequisites

Most of the models require a [Mujoco](https://www.roboti.us/license.html) license.


# Credits
The soft actor-critic algorithm was developed by Tuomas Haarnoja under the supervision of Prof. [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) and Prof. [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) at UC Berkeley. Special thanks to [Vitchyr Pong](https://github.com/vitchyr), who wrote some parts of the code, and [Kristian Hartikainen](https://github.com/hartikainen) who helped testing, documenting, and polishing the code and streamlining the installation process. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).

# Reference
```
@article{haarnoja2017soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={Deep Reinforcement Learning Symposium},
  year={2017}
}
```