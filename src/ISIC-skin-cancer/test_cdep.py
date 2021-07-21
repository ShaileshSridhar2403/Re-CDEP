import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import pdb

regularizer_rate=1


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        print("data",data)
        inputs,y,cd_features = data
        # cd_features = cd_features.eval(session=tf.compat.v1.Session())
        # pdb.set_trace()
        # cd_features = cd_features.numpy()
        # print("SHAPE", inputs.shape,y.shape)
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            add_loss = tf.zeros(1,)
            
            if regularizer_rate > 0:
            
                mask  = (cd_features[:, 0,0] != -1) #NOTE:CHECK
                # print("Mask",mask)
                # exit(0);
                # print("CD Features",cd_features)
                # if mask.any():
                if tf.experimental.numpy.any(mask):
                    rel, irrel = cd.cd_vgg_classifier(cd_features[:,0], cd_features[:,1], inputs, model)
                    cur_cd_loss = tf.nn.softmax(
                        tf.stack(
                                (
                                    # tf.boolean_mask(rel[:,0],mask),tf.boolean_mask(irrel[:,0],mask) NOTE: Cross Check with dimensions in original repo
                                    tf.boolean_mask(rel,mask),tf.boolean_mask(irrel,mask)
                                )
                                ,axis=1
                            )
                        )
                    cur_cd_loss = tf.nn.softmax(
                        tf.stack(
                            # (tf.boolean_mask(irrel[:,1],mask),tf.boolean_mask(rel[:,0],mask)),axis=1)
                            (tf.boolean_mask(irrel,mask),tf.boolean_mask(rel,mask)),axis=1
                        )
                    )
                    add_loss = cur_cd_loss/2
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            full_loss = loss+regularizer_rate*add_loss

            

            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(full_loss, trainable_vars)
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
    output_layer = tf.keras.layers.Dense(1,activation='sigmoid',name='predictions') #check if 4096 is just prev layer output or something more
    outputs = output_layer(x)
    model = CustomModel(classification_model.input,outputs)

    print(model.summary())
    return model

if __name__ == "__main__":
    model = create_classification_model()
    x = np.arange(25088).reshape(1,25088)
    x= tf.constant(x, dtype=tf.float64)

    y = np.arange(1).reshape(1,1)
    y = tf.constant(x,dtype=tf.float64)

    cd = np.arange(50176).reshape(2,25088)
    cd = tf.constant(cd,dtype=tf.float64)

    
    # model.predict((x,y,cd))
    pdb.set_trace()