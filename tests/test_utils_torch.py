import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh


def test_tensordataset():
    a = torch.FloatTensor([[1, 2], [3, 4], [5, 6]]).to('cuda')
    b = torch.FloatTensor([[7, 8], [9, 10], [11, 12]]).to('cuda')
    complete_dataset = torch.utils.data.TensorDataset(a, b)
    print(len(complete_dataset))

    num_train = int(len(complete_dataset)*.9)
    num_test = len(complete_dataset)  - num_train 
    torch.manual_seed(0);
    train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, [num_train, num_test])
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=1, shuffle=True) # create your dataloader
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #     batch_size=3, shuffle=True) # create your dataloader
    print(train_loader)
    # print(test_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        print(data)
        print(target)

test_tensordataset()
