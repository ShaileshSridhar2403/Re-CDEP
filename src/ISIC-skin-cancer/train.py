# import torch
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from torch import nn
import torch.optim as optim
# import torchvision.models as models
import sys
import pickle as pkl
import os
import argparse
from numpy.random import randint
import time
import copy
from tqdm import tqdm
sys.path.append('../../src')
import utils
import cd
import json
#torch.backends.cudnn.deterministic = True #this makes results reproducible.

# def cd_loss(inputs,labels,y_pred,cd_features):
    




with open('config.json') as json_file:
    data = json.load(json_file)
model_path = os.path.join(data["model_folder"], "ISIC_new")
dataset_path =os.path.join(data["data_folder"],"calculated_features")

 
# Training settings
parser = argparse.ArgumentParser(description='ISIC Skin cancer for CDEP')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--regularizer_rate', type=float, default=0.0, metavar='N',
                    help='hyperparameter for CDEP weight - higher means more regularization')
args = parser.parse_args()

regularizer_rate = args.regularizer_rate

num_epochs = args.epochs

# device = torch.device(0)

# load model
# torch.manual_seed(args.seed); #FIND WAY TO DO THIS IN KERAS
# model = models.vgg16(pretrained=True)

# model.classifier[-1] = nn.Linear(4096, 2)
# model = model.classifier.to(device)
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs,y,cd_features = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            # add_loss = torch.zeros(1,).cuda()
            add_loss = tf.zeros(1,)
            
            if regularizer_rate > 0:
            
                mask  = (cd_features[:, 0,0] != -1) #NOTE:CHECK
                if mask.any():
                    rel, irrel = cd.cd_vgg_classifier(cd_features[:,0], cd_features[:,1], inputs, model)
        
                    # cur_cd_loss = torch.nn.functional.softmax(torch.stack((rel[:,0].masked_select(mask),irrel[:,0].masked_select(mask)), dim =1), dim = 1)[:,0].mean() 
                    # cur_cd_loss +=torch.nn.functional.softmax(torch.stack((rel[:,1].masked_select(mask),irrel[:,1].masked_select(mask)), dim =1), dim = 1)[:,0].mean() 
                    cur_cd_loss = tf.nn.functional.softmax(tf.stack((tf.boolean_mask(rel[:,0],mask)),tf.boolean_mask(irrel[:,0],mask),axis=1))
                    cur_cd_loss = tf.nn.functional.softmax(tf.stack((tf.boolean_mask(irrel[:,1],mask)),tf.boolean_mask(rel[:,0],mask),axis=1))
                    add_loss = cur_cd_loss/2
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            full_loss = loss+regularizer_rate*add_loss

            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(full_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    



model = VGG16(weights='imagenet', include_top=False)
x = model.output
outputs = tf.keras.layers.Dense(2,activation='softmax') #check if 4096 is just prev layer output or something more
model = tf.keras.CustomModel(model.input,outputs)



datasets, weights = utils.load_precalculated_dataset(dataset_path)#NOTE: Not fully implemented, need to execute code to understand. REVISIT ALL DATASET/DATA LOADING CODE
if regularizer_rate ==-1: # -1 means that we train only on data with no patches
    datasets['train'] = datasets['train_no_patches']
    
dataset_sizes= {x:len(datasets[x]) for x in datasets.keys()}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in datasets.keys()}





# weights = torch.tensor(weights).to(device)

# params_to_update = model.parameters()

#optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
opt = tf.keras.optimizers.SGD(lr=args.lr, momentum=args.momentum)


# criterion = nn.CrossEntropyLoss(weight = weights.double().float())


model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
def train_model(model,dataloaders, num_epochs=25):
    since = time.time()
    val_acc_history = []
    val_loss_history = []
    train_loss_history = []
    
    train_acc_history = []
    train_cd_history= []
    model.fit(generator,epochs = num_epochs)

    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    patience = 3
    cur_patience = 0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_cd = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels, cd_features) in tqdm(enumerate(dataloaders[phase])):
    
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                cd_features = cd_features.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                    
                        add_loss = torch.zeros(1,).cuda()
                        if regularizer_rate > 0:
                        
                            mask  = (cd_features[:, 0,0] != -1).bool()
                            if mask.any():
                                rel, irrel = cd.cd_vgg_classifier(cd_features[:,0], cd_features[:,1], inputs, model)
                   
                                cur_cd_loss = torch.nn.functional.softmax(torch.stack((rel[:,0].masked_select(mask),irrel[:,0].masked_select(mask)), dim =1), dim = 1)[:,0].mean() 
                                cur_cd_loss +=torch.nn.functional.softmax(torch.stack((rel[:,1].masked_select(mask),irrel[:,1].masked_select(mask)), dim =1), dim = 1)[:,0].mean() 
                                add_loss = cur_cd_loss/2

                        (loss+regularizer_rate*add_loss).backward()
                        # print how much memory is used
                        #print(torch.cuda.memory_allocated()/(np.power(10,9)))
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_cd +=add_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_cd_loss = running_loss_cd / dataset_sizes[phase]
       
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
  
            
            print('{} Loss: {:.4f} Acc: {:.4f} CD Loss : {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_cd_loss))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_cd_history.append(epoch_cd_loss)
                train_acc_history.append(epoch_acc.item())
                
            if phase == 'val':
                if epoch_loss < best_loss:
            
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    cur_patience = 0
                else:
                    cur_patience+=1
        if cur_patience >= patience:
            break
             
                

    '''

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # print('Best val loss: {:4f}'.format(best_loss))


  
    hist_dict = {}
    hist_dict['val_acc_history'] = val_acc_history
    hist_dict['val_loss_history'] = val_loss_history
    hist_dict['train_acc_history'] = train_acc_history
    hist_dict['train_loss_history'] = val_loss_history
    hist_dict['train_cd_history'] = train_cd_history
    model.load_state_dict(best_model_wts)
    return model,hist_dict 
    


model, hist_dict = train_model(model, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

hist_dict['AUC (no patches)'],hist_dict['F1 score (no patches)'] =utils.get_auc_f1(model, datasets['test_no_patches'])

hist_dict['AUC (patches)'],hist_dict['F1 score (patches)'] =utils.get_auc_f1(model, datasets['test'])



pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
#torch.save(model.state_dict(),oj(model_path, pid + ".pt"))

hist_dict['pid'] = pid
hist_dict['regularizer_rate'] = regularizer_rate
hist_dict['seed'] = args.seed
hist_dict['batch_size'] = args.batch_size
hist_dict['learning_rate'] = args.lr
hist_dict['momentum'] = args.momentum
pkl.dump(hist_dict, open(os.path.join(model_path , pid +  '.pkl'), 'wb'))