"""test-transformers: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import tensorflow as tf
from typing import Dict
import os
import pandas as pd

from model_generation.example_mnist.model_definition import _get_model as get_model

# Define the custom strategy with cumulative metrics tracking
class CustomStrategy(FedAvg):
    def __init__(
        self,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=None,
        min_clients=1,
        config={},
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=initial_parameters,
            min_fit_clients=min_clients,
        )
        self.config = config
        self.energy_sum = 0
        self.time_sum = 0

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregates fit results, logs metrics for each round, and appends them to a CSV file.

        :param rnd: Current round number.
        :param results: Results from clients.
        :param failures: Failed client results (not used here but included for compatibility).
        """
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)
        result_path = self.config["simulation_result_path"]
        result_df = get_result_df(result_path)

        for _, client_result in results:
            metrics = client_result.metrics
            new_row = {
                "round": rnd,
                "energy": metrics.get("energy", 0),
                "time": metrics.get("time", 0),
                "epochs": metrics.get("epochs", 0),
            }
            result_df = pd.concat(
                [result_df, pd.DataFrame([new_row])], ignore_index=True
            )
            print(result_df)
            # Accumulate specific metrics
            self.energy_sum += new_row["energy"]
            self.time_sum += new_row["time"]

        # Save the updated DataFrame to the CSV file
        result_df.to_csv(result_path, index=False)
        print(f"After round {rnd}, energy: {self.energy_sum}, time: {self.time_sum}")

        return aggregated_weights, {"energy sum": self.energy_sum}

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(rnd, results, failures)
        print(f"After round {rnd}, loss: {aggregated_loss}")
        return aggregated_loss, {"energy sum": self.energy_sum}


# Define server function to initialize the server with a custom strategy
def server_fn(context: Context, config: Dict):
    # Server configuration
    num_rounds = config.get("num_rounds", 1)
    fraction_fit = config.get("fraction_fit", 1)

    # Initialize the model and get initial weights
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    initial_weights = model.get_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Create a custom strategy with the initial parameters
    strategy = CustomStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        initial_parameters=initial_parameters,
        min_clients=config.get("min_client_per_round", 1),
        config=config
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


def create_server_app(config):
    return ServerApp(server_fn=lambda context: server_fn(context, config))


def get_result_df(result_path):
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
    else:
        df = pd.DataFrame(columns=["round", "energy", "time"])
    return df
