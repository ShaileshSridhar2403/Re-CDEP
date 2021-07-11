import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
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
from config import config as data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc

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




class CustomModel(tf.keras.Model):
    def train_step(self, data):
        inputs,y,cd_features = data
        print("SHAPE", inputs.shape,y.shape)
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            

            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


    def test_step(self,data):
        inputs,y,cd_features = data
        y_pred = self(inputs, training=False)

        #NOTE: Just using binary cross entropy loss for now, might need to revisit and get a loss similar to torch code
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def create_classification_model():
    vgg = VGG16(weights='imagenet', include_top=True)

    classification_model = tf.keras.Sequential()

    classification_model.add(tf.keras.layers.Dense(4096,input_shape=(25088,),activation='relu',name='fc1'))
    classification_model.layers[0].set_weights(vgg.get_layer('fc1').get_weights())

    classification_model.add(tf.keras.layers.Dense(4096,activation='relu',name='fc2'))
    classification_model.layers[1].set_weights(vgg.get_layer('fc2').get_weights())

    x = classification_model.output
    output_layer = tf.keras.layers.Dense(1,activation='sigmoid') #check if 4096 is just prev layer output or something more
    outputs = output_layer(x)
    model = CustomModel(classification_model.input,outputs)

    print(model.summary())
    return model





model = create_classification_model()


datasets, weights = utils.load_precalculated_dataset(dataset_path)#NOTE: Not fully implemented, need to execute code to understand. REVISIT ALL DATASET/DATA LOADING CODE


if regularizer_rate ==-1: # -1 means that we train only on data with no patches
    datasets['train'] = datasets['train_no_patches']

try:
    dataset_sizes = {}
    for x in datasets:
        dataset_sizes[x] = len(list(datasets[x].as_numpy_iterator()))

except:
    print("\n error_set",x,"\n",datasets[x])


opt = tf.keras.optimizers.SGD(lr=args.lr, momentum=args.momentum)



#NOTE: Check if loss should be sparse categorical
#NOTE2: Changed loss from categorical crossentropy, due to this issue https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible:
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
def train_model(model,train_dataset,val_dataset, num_epochs=25):
    since = time.time()
    
    H = model.fit(train_dataset,validation_data= val_dataset,epochs = num_epochs)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

    #NOTE: Did not use patience or best model like in original repo
    return model,{}
    

train_dataset = datasets['train'].batch(args.batch_size)


model, hist_dict = train_model(model, train_dataset, val_dataset = datasets['val'].batch(args.batch_size),num_epochs=num_epochs)

val_y = [y for x, y,z in datasets['val']]
test_y = [y for x,y,z in datasets['test']]

val_predictions = [int(y>0.5) for y in model.predict(datasets["val"].batch(args.batch_size))]
test_predictions = [int(y>0.5) for y in model.predict(datasets["test"].batch(args.batch_size))]

val_prediction_probs = [y for y in model.predict(datasets["val"].batch(args.batch_size))]
test_prediction_probs = [y for y in model.predict(datasets["test"].batch(args.batch_size))]


print("Printing Validation Scores............")
print(classification_report(val_y,
	val_predictions, target_names=["Not Cancer","Cancer"]))

print("Printing Test Scores.........")
print(classification_report(test_y,
    test_predictions, target_names=["Not Cancer","Cancer"]))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y,test_prediction_probs)
auc_output = auc(fpr_keras,tpr_keras)

print(f"\nTest AUC {auc_output}")




pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])

hist_dict['pid'] = pid
hist_dict['regularizer_rate'] = regularizer_rate
hist_dict['seed'] = args.seed
hist_dict['batch_size'] = args.batch_size
hist_dict['learning_rate'] = args.lr
hist_dict['momentum'] = args.momentum