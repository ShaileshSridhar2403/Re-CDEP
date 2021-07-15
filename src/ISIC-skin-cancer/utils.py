# from torch.utils.data import TensorDataset, ConcatDataset
import tensorflow as tf
import numpy as np
# import torch.utils.data as utils
# from torch.utils.data import DataLoader
from sklearn.metrics import auc,average_precision_score, roc_curve,roc_auc_score,precision_recall_curve, f1_score
# from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
# import torchvision
# from torchvision.transforms import ToTensor, Compose, Normalize
# import torch
import os

def check_and_convert_to_NHWC(img):
    print(img.shape)
    if img.shape[-1] != 3:
        img = tf.transpose(img, [0, 2, 3, 1])
        print("converted shape",img.shape)
    return img

def filter_dataset(x,y,z):
    # return z[0] == -1
    return True

def calc_weights(num_cancer, num_complete):
    cancer_ratio =num_cancer/num_complete


    not_cancer_ratio = 1- cancer_ratio
    cancer_weight = 1/cancer_ratio
    not_cancer_weight = 1/ not_cancer_ratio
    weights = np.asarray([not_cancer_weight, cancer_weight])
    weights /= weights.sum()
    weights_dict = {
        0: weights[0],
        1: weights[1]
    }
    return weights
        
def load_precalculated_dataset(path):
    with open(os.path.join(path, "cancer.npy"), 'rb') as f:
        cancer_features = np.load(f)
    with open(os.path.join(path, "not_cancer.npy"), 'rb') as f:
        not_cancer_features = np.load(f)
    with open(os.path.join(path, "not_cancer_cd.npy"), 'rb') as f:
        not_cancer_cd= np.load(f)  


    print("not cancer cd",not_cancer_cd)
    print("cancer",cancer_features)
    print("not_cancer_cd",not_cancer_features);
    # exit(0)

    cancer_targets = np.ones((cancer_features.shape[0])).astype(np.int64)
    not_cancer_targets = np.zeros((not_cancer_features.shape[0])).astype(np.int64)

    
    # not_cancer_dataset = TensorDataset(torch.from_numpy(not_cancer_features).float(), torch.from_numpy(not_cancer_targets),torch.from_numpy(not_cancer_cd).float())
    not_cancer_dataset = tf.data.Dataset.from_tensor_slices((not_cancer_features,not_cancer_targets,not_cancer_cd))
    cancer_dataset = tf.data.Dataset.from_tensor_slices((cancer_features, cancer_targets,-np.ones((len(cancer_features), 2, 25088))))
    # complete_dataset = ConcatDataset((not_cancer_dataset,cancer_dataset ))
    complete_dataset = not_cancer_dataset.concatenate(cancer_dataset)
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    num_test = num_total - num_train - num_val
    # torch.manual_seed(0); #reproducible splitting
    # train_dataset, test_dataset, val_dataset= torch.utils.data.random_split(complete_dataset, [num_train, num_test, num_val])
    # train_dataset,test_dataset,val_dataset = tf.split(complete_dataset, [num_train, num_test, num_val])
    complete_dataset = complete_dataset.shuffle(len(complete_dataset),seed=42)
    train_dataset = complete_dataset.take(num_train)
    test_dataset = complete_dataset.skip(num_train).take(num_test)
    val_dataset = complete_dataset.skip(num_train).skip(num_test).take(num_val)

    #NOTE: Leave for now, revisit when possible to execute
    #NOTE 2: revisited,coded, length of non filtered was 0 in all cases still need to underestand what is going on

    # train_filtered_dataset =torch.utils.data.Subset(complete_dataset, [idx for idx in train_dataset.indices if complete_dataset[idx][2][0,0] ==-1])
    # test_filtered_dataset = torch.utils.data.Subset(complete_dataset, [idx for idx in test_dataset.indices if complete_dataset[idx][2][0,0] ==-1])
    # val_filtered_dataset = torch.utils.data.Subset(complete_dataset, [idx for idx in test_dataset.indices if complete_dataset[idx][2][0,0] ==-1])

    # train_filtered_dataset = train_dataset.filter(lambda x: x[2][0,0] == -1)
    # test_filtered_dataset = test_dataset.filter(lambda x: x[2][0,0] == -1)
    # val_filtered_dataset = val_dataset.filter(lambda x: x[2][0,0] == -1)

    train_filtered_dataset = train_dataset.filter(filter_dataset)
    test_filtered_dataset = test_dataset.filter(filter_dataset)
    val_filtered_dataset = val_dataset.filter(filter_dataset)
    # print("LENGTH",list(val_filtered_dataset.as_numpy_iterator()))

    # datasets = {'train': train_dataset,'train_no_patches': train_filtered_dataset, 'val':val_dataset ,'val_no_patches':val_filtered_dataset ,'test':test_dataset, 'test_no_patches':test_filtered_dataset }
    datasets = {'train':train_dataset,'train_no_patches':train_filtered_dataset,'val':val_dataset,'val_no_patches':val_filtered_dataset,'test':test_dataset,'test_no_patches':test_filtered_dataset}
    return datasets, calc_weights(len(cancer_dataset), len(complete_dataset))

    
def get_output(model, dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                             shuffle=False, num_workers=4)
    model = model.eval()
    y = []
    y_hat = []
    softmax= torch.nn.Softmax()
    with torch.no_grad() :
        for inputs, labels, cd in data_loader:
            y_hat.append((labels).cpu().numpy())
            y.append(torch.nn.Softmax(dim=1)( model(inputs.cuda()))[:,1].detach().cpu().numpy()) # take the probability for cancer
    y_hat = np.concatenate( y_hat, axis=0 )
    y = np.concatenate( y, axis=0 )
    return y, y_hat # in the training set the values were switched
    
    


def get_auc_f1(model, dataset,fname = None, ):
    if fname !=None:
        with open(fname, 'rb') as f:
            weights = torch.load(f)
        if "classifier.0.weight" in weights.keys(): #for the gradient models we unfortunately saved all of the weights
            model.load_state_dict(weights)
        else:
            model.classifier.load_state_dict(weights)
        y, y_hat = get_output(model.classifier, dataset)
    else:   
        y, y_hat = get_output(model, dataset)
    auc =roc_auc_score(y_hat, y)
    f1 = np.asarray([f1_score(y_hat, y > x) for x in np.linspace(0.1,1, num = 10) if (y >x).any() and (y<x).any()]).max()
    return auc, f1

def load_img_dataset(path):

    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    complete_dataset = torchvision.datasets.ImageFolder(path, transform=Compose([ToTensor(), Normalize(mean, std)]))
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    num_test = num_total - num_train - num_val
    torch.manual_seed(0);
    train_dataset, test_dataset, val_dataset= torch.utils.data.random_split(complete_dataset, [num_train, num_test, num_val])
    return test_dataset  