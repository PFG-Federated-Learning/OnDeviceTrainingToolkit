"""test-transformers: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import tensorflow as tf
from typing import Dict

from src.task_test import get_model

# Define the custom strategy with cumulative metrics tracking
class CustomStrategy(FedAvg):
    def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0, initial_parameters=None):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=initial_parameters,
        )
        self.energy_sum = 0

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)
        
        # Accumulate metrics for each round
        for _, client_result in results:
            for metric_name, metric_value in client_result.metrics.items():
                if metric_name == "energy":
                    self.energy_sum+=metric_value
        print(f"After round {rnd}, energy: {self.energy_sum}")
        
        return aggregated_weights, {"energy sum": self.energy_sum}

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(rnd, results, failures)
        print(f"After round {rnd}, loss: {aggregated_loss}")
        return aggregated_loss, {"energy sum": self.energy_sum}

# Define server function to initialize the server with a custom strategy
def server_fn(context: Context, config: Dict):
    # Server configuration
    num_rounds = config.get("num_rounds", 3)
    fraction_fit = config.get("fraction_fit", 0.5)

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
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def create_server_app(config):
    return ServerApp(server_fn=lambda context: server_fn(context, config))