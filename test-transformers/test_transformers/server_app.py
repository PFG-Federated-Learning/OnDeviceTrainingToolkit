"""test-transformers: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForAudioClassification

from test_transformers.task import get_weights


def server_fn(context: Context, config):
    # Read from config
    num_rounds = 3
    fraction_fit = 0.5

    # Initialize global model
    model_name = config["model"]
    num_labels = 2
    net = AutoModelForAudioClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


def create_server_app(config):
    return ServerApp(server_fn=lambda context: server_fn(context, config))