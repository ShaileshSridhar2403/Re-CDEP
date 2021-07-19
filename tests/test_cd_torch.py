from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch import tanh

stabilizing_constant = 10e-20


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    def logits(self, x):
    
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

blob = np.zeros((28,28))
size_blob =5
blob[:size_blob, :size_blob ] =1

blob[-size_blob:, :size_blob] = 1
blob[:size_blob, -size_blob: ] =1
blob[-size_blob:, -size_blob:] = 1

# propagate convolutional or linear layer
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

# propagate ReLu nonlinearity
def propagate_relu(relevant, irrelevant, activation, device='cuda'):
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    zeros = torch.zeros(relevant.size()).to(device)
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score

# propagate maxpooling operation
def propagate_pooling(relevant, irrelevant, pooler, model_type='mnist'):
    if model_type == 'mnist':
        unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        window_size = 4
    elif model_type == 'vgg':
        unpool = torch.nn.MaxUnpool2d(kernel_size=pooler.kernel_size, stride=pooler.stride)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=(pooler.kernel_size, pooler.kernel_size),
                                        stride=(pooler.stride, pooler.stride), count_include_pad=False)
        window_size = 4

    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)
    ones_out = torch.ones_like(both)
    size1 = relevant.size()
    mask_both = unpool(ones_out, both_ind, output_size=size1)

    # relevant
    rel = mask_both * relevant
    rel = avg_pooler(rel) * window_size

    # irrelevant
    irrel = mask_both * irrelevant
    irrel = avg_pooler(irrel) * window_size
    return rel, irrel

# propagate dropout operation
def propagate_dropout(relevant, irrelevant, dropout):
    return dropout(relevant), dropout(irrelevant)

# get contextual decomposition scores for blob
def cd(blob, im_torch, model, model_type='mnist', device='cuda'):
 
    # set up model
    model.eval()
    im_torch = im_torch.to(device)
    
    # set up blobs
    blob = torch.FloatTensor(blob).to(device)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch
  

    if model_type == 'mnist':
        scores = []
        mods = list(model.modules())[1:]
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
 
     

        relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                                 lambda x: F.max_pool2d(x, 2, return_indices=True), model_type='mnist')
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[1])

        relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                                 lambda x: F.max_pool2d(x, 2, return_indices=True), model_type='mnist')
 
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)
     
        relevant = relevant.view(-1, 800)
        irrelevant = irrelevant.view(-1, 800)
 

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])
    
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu) 
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[3])
    

    else:
        mods = list(model.modules())
        for i, mod in enumerate(mods):
            t = str(type(mod))
            if 'Conv2d' in t or 'Linear' in t:
                if 'Linear' in t:
                    relevant = relevant.view(relevant.size(0), -1)
                    irrelevant = irrelevant.view(irrelevant.size(0), -1)
                relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
            elif 'ReLU' in t:
                relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
            elif 'MaxPool2d' in t:
                relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod, model_type=model_type)
            elif 'Dropout' in t:
                relevant, irrelevant = propagate_dropout(relevant, irrelevant, mod)
    return relevant, irrelevant


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
    # a = torch.FloatTensor([[[[4., 5, 6, 7],
    #                     [8, 9, 10, 11],
    #                     [12, 13, 14, 15],
    #                     [16, 17, 18, 19]]]])
    # x = np.arange(2352).reshape(3, 1, 28, 28)
    x = np.arange(18).reshape(2, 1, 3, 3)
    # x = np.arange(16*6*6*512).reshape(16, 512, 6, 6)
    # x = np.arange(16*8*8*512).reshape(16, 512, 8, 8)
    a = torch.from_numpy(x).float().to('cuda')

    b, b_ind = F.max_pool2d(a, 2, return_indices=True)

    unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
    c = unpool(b, b_ind)

    print(c)

def test_propagate_dropout():
    a = torch.FloatTensor([1.0, 2.0, 3.0, 4.0]).to('cuda')
    b = torch.FloatTensor([1.0, 2.0, 3.0, 4.0]).to('cuda')

    c = torch.nn.Dropout()
    result = propagate_dropout(a, b, c)

    print(result)

def test_propagate_pooling():
    # a = torch.FloatTensor([[[[4., 5, 6, 7],
    #                     [8, 9, 10, 11],
    #                     [12, 13, 14, 15],
    #                     [16, 17, 18, 19]]]]).to('cuda')
    # b = torch.FloatTensor([[[[1., 2, 3, 4],
    #                     [5, 6, 7, 8],
    #                     [9, 10, 11, 12],
    #                     [13, 14, 15, 16]]]]).to('cuda')

    x = np.arange(2352).reshape(3, 1, 28, 28)
    y = np.arange(2352).reshape(3, 1, 28, 28)
    a = torch.from_numpy(x).float().to('cuda')
    b = torch.from_numpy(y).float().to('cuda')

    c = lambda x: F.max_pool2d(x, 2, return_indices=True)
    result = propagate_pooling(a, b, c)

    print(result)

def test_cd():
    model = Net()
    model.cuda()

    # float64 is double
    x = np.arange(1568).reshape(2, 1, 28, 28)
    # x = np.arange(784).reshape(1, 1, 28, 28)
    a = torch.from_numpy(x).to('cuda')
    # a = tf.constant(x, dtype=tf.float64)
    result = cd(blob=blob, im_torch=a, model=model)
    print(result)

test_unpool()
