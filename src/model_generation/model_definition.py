import tensorflow as tf
from constants import *

def _get_model():
    '''Define your model's backbone in this function'''

    # Below is a sample backbone of a simple MNIST model
    
    # In the input layer the batch_size argument MUST be given for the model
    # to work in the app
    inputs = tf.keras.Input(shape=FEATURE_SHAPE, batch_size=BATCH_SIZE, dtype=tf.float32)
    net = tf.keras.layers.Conv2D(16, [3, 3])(inputs)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(128, activation='relu', name='dense_1', dtype=tf.float32)(net)
    out = tf.keras.layers.Dense(10, name='dense_2', dtype=tf.float32)(net)

    model = tf.keras.models.Model(inputs, out)

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