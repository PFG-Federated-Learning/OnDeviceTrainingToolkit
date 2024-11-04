"""test-transformers: A Flower / HuggingFace app."""

from random import randint
from flwr.client import ClientApp

import tensorflow as tf
import flwr as fl

from src.task_test import get_ds, get_model, get_processed_ds

# Define a Flower client for each simulated client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, client_metrics):
        self.model = model
        self.train_ds = train_ds
        self.epochs = client_metrics["epochs"]
        self.energy = client_metrics["energy"]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_ds, epochs=self.epochs, verbose=0)
        return self.model.get_weights(), len(self.train_ds), {"energy": self.energy}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.train_ds, verbose=0)
        return loss, len(self.train_ds), {"accuracy": accuracy}

# Flower client function
def client_fn(context, config):
    # Load and preprocess the dataset for each client
    train_ds, _, ds_info = get_ds()
    train_ds = get_processed_ds(train_ds, ds_info)

    # Instantiate and return the client
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    clients = config["clients"]
    client_metrics = clients[randint(0,len(clients)-1)]
    return MnistClient(model, train_ds, client_metrics).to_client()
def create_client_app(config):
    return ClientApp(client_fn=lambda context: client_fn(context, config))