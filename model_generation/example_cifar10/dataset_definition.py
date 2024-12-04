import tensorflow as tf
import tensorflow_datasets as tfds

from model_generation.constants import *

# This file should contain the pipeline to obtain your dataset

# You can alter the following function as desired
# In the following code, as example, we load and pre-process
# the MNIST dataset

NUM_SAMPLES = 1024


def get_processed_ds():
    cifar10 = tf.keras.datasets.cifar10

    # Distribute it to train and test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Reduce pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # flatten the label values
    y_train, y_test = y_train.flatten(), y_test.flatten()

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        new_label = tf.one_hot(indices=label, depth=10)
        return tf.cast(image, tf.float32), tf.cast(new_label, tf.float32)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES])
    )
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.batch(BATCH_SIZE)
    # number of classes
    K = len(set(y_train))

    return ds_train, NUM_SAMPLES
