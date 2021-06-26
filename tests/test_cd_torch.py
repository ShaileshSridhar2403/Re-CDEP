import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch import tanh

stabilizing_constant = 10e-20


def propagate_conv_linear(relevant, irrelevant, module, device='cuda'):
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel + stabilizing_constant

    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


def test_propagate_conv_linear():
    a = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]).to('cuda')
    b = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]).to('cuda')
    # c = torch.FloatTensor([[1.0, 1.0], [0.0, 1.0]]).to('cuda')

    l = nn.Linear(2, 2).to('cuda')
    net = nn.Sequential(l).to('cuda')
    mods = list(net.modules())[1:]
    mods[0].weight.data = torch.ones(mods[0].weight.size()).to('cuda')
    mods[0].bias.data = torch.ones(mods[0].bias.size()).to('cuda')

    print(mods[0].weight)
    print(mods[0].bias)

    r1, r2 = propagate_conv_linear(a, b, mods[0])
    print(r1)
    print()
    print(r2)
    # print()
    # print(r3)

def test_unpool():
    a = torch.FloatTensor([[[[4., 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15],
                        [16, 17, 18, 19]]]])
    print(a)

    b, b_ind = F.max_pool2d(a, 2, return_indices=True)
    print(b)
    print()
    print(b_ind)

    unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
    print(unpool)


test_propagate_conv_linear()
