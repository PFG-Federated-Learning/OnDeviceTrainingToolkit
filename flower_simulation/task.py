import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import flwr as fl
from typing import Dict, Tuple

IMG_SIZE = 28


# Define the model as before
def get_model():
    inputs = tf.keras.Input(shape=[IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32)
    net = tf.keras.layers.Conv2D(16, [3, 3])(inputs)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(
        128, activation="relu", name="dense_1", dtype=tf.float32
    )(net)
    out = tf.keras.layers.Dense(10, name="dense_2", dtype=tf.float32)(net)
    return tf.keras.models.Model(inputs, out)


# Load and process dataset as before
def get_ds():
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_test, ds_info


def get_processed_ds(ds_train, ds_info):
    def normalize_img(image, label):
        new_label = tf.one_hot(indices=label, depth=10)
        return tf.cast(image, tf.float32) / 255.0, tf.cast(new_label, tf.float32)

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    return ds_train
