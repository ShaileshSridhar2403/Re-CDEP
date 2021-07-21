import tensorflow as tf
import pdb
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import sys
sys.path.append("../src")
import cd


class CustomModel(tf.keras.Model):
    def __init__(self, model, decoy, signal_strength):
        super(CustomModel, self).__init__()
        self.decoy = decoy
        self.model = model
        self.signal_strength = signal_strength
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.c
        
        text, labels = data

        with tf.GradientTape() as tape:
            y_pred = self.model(text, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            
            loss = self.compiled_loss(labels, y_pred, regularization_losses=self.losses)
            #loss = self.loss_fn(labels, y_pred)
            
            #start = (( text==self.decoy[0]) + (text == self.decoy[1])).double().argmax(dim = 0) 
            #start[(( text ==self.decoy[0]) + (text == self.decoy[1])).sum(dim=0) ==0] = -1 # if there is none, set to -1
            
            if self.signal_strength > 0:
                start = tf.argmax(tf.cast(text==self.decoy[0], tf.float64) + tf.cast(text==self.decoy[0], tf.float64), axis=1)
                mask = tf.reduce_sum((tf.cast(text==self.decoy[0], tf.float64) + tf.cast(text==self.decoy[0], tf.float64)), axis=1)
                start = tf.cast(start, dtype=tf.int32)
                start = tf.where(mask == 0, -1, start)
                stop = start + 1 
                #pdb.set_trace()
            
            
                cd_loss = cd.cd_penalty_for_one_decoy_all(text, labels, self.model, start, stop) 
                #print(cd_loss.data.item()/ total_loss.data.item())
                total_loss = loss+ self.signal_strength*cd_loss
                #print(f'loss:{loss} cd_loss:{cd_loss} Total Loss:{total_loss}')
            else:
                total_loss = loss
        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        #pdb.set_trace()
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, y_pred)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        if self.signal_strength > 0:
            metrics['cd_loss'] = cd_loss
        metrics['Total_loss'] = total_loss
        return metrics
    
    def test_step(self,data):
        text, labels = data
        y_pred = self.model(text, training=False)

        loss = self.compiled_loss(labels, y_pred, regularization_losses=self.losses)

        if self.signal_strength > 0:
            start = tf.argmax(tf.cast(text==self.decoy[0], tf.float64) + tf.cast(text==self.decoy[0], tf.float64), axis=1)
            mask = tf.reduce_sum((tf.cast(text==self.decoy[0], tf.float64) + tf.cast(text==self.decoy[0], tf.float64)), axis=1)
            start = tf.cast(start, dtype=tf.int32)
            start = tf.where(mask == 0, -1, start)
            stop = start + 1 
                #pdb.set_trace()
            
            
            cd_loss = cd.cd_penalty_for_one_decoy_all(text, labels, self.model, start, stop) 
            #print(cd_loss.data.item()/ total_loss.data.item())
            total_loss = loss+ self.signal_strength*cd_loss
            #print(f'loss:{loss} cd_loss:{cd_loss} Total Loss:{total_loss}')
        else:
            total_loss = loss
        self.compiled_metrics.update_state(labels, y_pred)

        metrics = {m.name: m.result() for m in self.metrics}
        metrics['total_val_loss'] = total_loss
        return metrics


def build_model():
    inputs = Input(shape=(None,))
    x = Embedding(input_dim=1000, output_dim=300)(inputs)
    x = LSTM(units=128)(x)
    outputs = Dense(2)(x)
    model = CustomModel(inputs=inputs, outputs=outputs, decoy='a')
    return model


class LSTMSentiment(tf.keras.Model):

    def __init__(self):
        super(LSTMSentiment, self).__init__()
        self.embedding = Embedding(input_dim=14805+1, output_dim=300)
        self.lstm = LSTM(units=128)
        self.outputs = Dense(2)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return self.outputs(x)


