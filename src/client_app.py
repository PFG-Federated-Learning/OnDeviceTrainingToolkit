"""test-transformers: A Flower / HuggingFace app."""

from random import randint
from flwr.client import ClientApp

import tensorflow as tf
import flwr as fl

from model_generation.example_cifar10.model_definition import _get_model as get_model
from model_generation.example_cifar10.dataset_definition import get_processed_ds

# Define a Flower client for each simulated client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, ds_samples,client_configs):
        self.model = model
        self.train_ds = train_ds
        self.ds_samples = ds_samples
        self.epochs = client_configs["epochs"]
        self.energy_per_sample = client_configs["sampleEnergy"]
        self.time_per_sample = client_configs["sampleTime"]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        energy = self.energy_per_sample * self.ds_samples
        time = self.time_per_sample * self.ds_samples
        self.model.fit(self.train_ds, epochs=self.epochs, verbose=0)
        return self.model.get_weights(), len(self.train_ds), {"energy": energy, "time": time}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.train_ds, verbose=0)
        return loss, len(self.train_ds), {"accuracy": accuracy}

def client_fn(context, config):
    train_ds, ds_samples = get_processed_ds()
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    clients = config["devices_config"]
    client_configs = clients[randint(0,len(clients)-1)]
    return MnistClient(model, train_ds, ds_samples, client_configs).to_client()

def create_client_app(config):
    return ClientApp(client_fn=lambda context: client_fn(context, config))