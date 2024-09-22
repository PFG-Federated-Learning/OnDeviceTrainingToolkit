import logging
from omegaconf import DictConfig
import hydra
from server import create_server_app
from client import create_client_app
import torch
from flwr.simulation import run_simulation


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
log = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_gpus": 1}}

    NUM_PARTITIONS = 10

    # Run simulation
    run_simulation(
        server_app=create_server_app(),
        client_app=create_client_app(),
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )

if __name__ == "__main__":
    my_app()
