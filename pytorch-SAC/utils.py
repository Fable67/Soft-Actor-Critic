import torch


def hard_copy(targ_net, net):
    for p_target, p in zip(targ_net.parameters(), net.parameters()):
        p_target.data.copy_(p.data)
    return targ_net


def soft_copy(targ_net, net, tau):
    for target_p, p in zip(targ_net.parameters(), net.parameters()):
        target_p.data.copy_(target_p.data * (1. - tau) + p.data * tau)
    return targ_net
