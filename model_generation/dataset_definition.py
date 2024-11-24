import tensorflow as tf
import tensorflow_datasets as tfds

from constants import *

# This file should contain the pipeline to obtain your dataset

# You can alter the following function as desired
# In the following code, as example, we load and pre-process 
# the MNIST dataset

def get_processed_ds():

    def get_ds():
        (ds_train, ds_test), ds_info = tfds.load(
                'mnist',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )
        return ds_train, ds_test, ds_info

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        new_label = tf.one_hot(indices=label, depth=10)
        return tf.cast(image, tf.float32) / 255., tf.cast(new_label, tf.float32)
   
    ds_train, ds_test, ds_info = get_ds()

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)

    return ds_train