from model_definition import MyModel
from dataset_definition import get_processed_ds
from constants import *

from tqdm import tqdm
import tensorflow as tf
import numpy as np


def train_model(model: MyModel, ds_train):
    """
    Trains the given model using ds_train and saves the features and labels
    from the dataset to a binary file.
    Parameters:
        model: The model to be trained, instance of MyModel
        ds_train: iterable representing the dataset to train the model.
                  Its items should be pairs (x, y) of (features, labels)
    """
    x_all = None
    y_all = None
    curr_loss = float("inf")

    # Training loop - the model should be initialized with a local training
    for x, y in (pbar := tqdm(ds_train, desc=f"Training: loss = {curr_loss}")):
        # x_all and y_all are numpy arrays that will contain all the features
        # so that they can be saved to binary files later
        if x_all is None:
            x_all = np.array(x)
            y_all = np.array(y)
        else:
            x_all = np.append(x_all, x, axis=0)
            y_all = np.append(y_all, y, axis=0)

        # Perform training step
        curr_loss = model.train(x, y)["loss"]
        pbar.set_description(f"Training: loss = {curr_loss}")

    # Save the features to a binary format
    x_all.tofile(FEATS_BIN_FILE)
    y_all.tofile(LABELS_BIN_FILE)

    return model


def save_keras_model(model, model_dir):
    # Save the Keras model
    # The "signatures" saved in this step are the functions that will
    # be available to be called on device
    tf.saved_model.save(
        model,
        model_dir,
        signatures={
            "train": model.train.get_concrete_function(),
            "infer": model.infer.get_concrete_function(),
            "save": model.save.get_concrete_function(),
            "restore": model.restore.get_concrete_function(),
        },
    )


def save_tflite_model(saved_model_dir):
    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable LiteRT ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter.allow_custom_ops = True
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # Save the TFLite model to file
    with open(TFLITE_MODEL_FILE, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def main():
    model = MyModel()
    ds_train = get_processed_ds()

    model = train_model(model, ds_train)

    save_keras_model(model, SAVED_MODEL_DIR)

    tflite_model = save_tflite_model(SAVED_MODEL_DIR)

    # Test the converted model by running inference and comparing with keras model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    infer = interpreter.get_signature_runner("infer")

    sample_batch = np.array([x for (x, _) in ds_train.take(1).as_numpy_iterator()][0])
    print("\nKERAS OUTPUT:\n", model.infer(sample_batch)["logits"][0])
    print(
        "TFLITE OUTPUT:\n",
        infer(x=np.array(sample_batch).astype(np.float32))["logits"][0],
    )


if __name__ == "__main__":
    main()
