import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from model_generation.example_cifar10.constants import *

def _get_model():
    '''Define your model's backbone in this function'''

    # Below is a sample backbone of a simple MNIST model
    
    # In the input layer the batch_size argument MUST be given for the model
    # to work in the app
    # Build the model using the functional API
    # input layer
    i = Input(shape=FEATURE_SHAPE, batch_size=BATCH_SIZE, dtype=tf.float32)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), )(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)

    # Hidden layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    # last hidden layer i.e.. output layer
    x = Dense(OUTPUT_SHAPE[0])(x)

    model = Model(i, x)

    return model

class MyModel(tf.Module):
    '''
    This is the class that defines the actual model that will be exported to
    the app. 
    Besides from the backbone defined in the get_model function, it also 
    contains the functions that the model will be able to call in the app.
    '''

    def __init__(self):
        self.model = _get_model()
        self.model.summary()

        # You may change the optimizer and loss as desired
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
        tf.TensorSpec([BATCH_SIZE] + FEATURE_SHAPE, tf.float32),
        tf.TensorSpec([BATCH_SIZE] + OUTPUT_SHAPE, tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            result = {"loss": loss}
            return result

    @tf.function(input_signature=[
        tf.TensorSpec([BATCH_SIZE] + FEATURE_SHAPE, tf.float32),
    ])
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors