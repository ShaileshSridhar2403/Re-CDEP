import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_nll_loss():
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    print(x.shape)
    a = torch.from_numpy(x).to('cuda')
    # print(F.log_softmax(a))
    target = torch.tensor([1, 0, 4]).to('cuda')
    output = F.nll_loss(nn.log_softmax(a), target).item()
    print(output)

test_nll_loss()
