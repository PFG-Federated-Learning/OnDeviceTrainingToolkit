from flwr.client import Client, ClientApp, NumPyClient
from model import get_parameters, set_parameters, train, test, Net
from flwr.common import Context
from dataset import load_datasets
import torch
import hydra
from random import randint

class FlowerNumPyClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, time_to_process):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.time_to_process = time_to_process
        self.total_time_to_process = 0

    def get_time_to_process(self):
        print(f"[Client {self.partition_id}] time to process: {self.time_to_process}")
        self.total_time_to_process += self.time_to_process
        print(f"[Client {self.partition_id}] total time to process: {self.total_time_to_process}")


    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        self.get_time_to_process()
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



def numpyclient_fn(context: Context, clients: list) -> Client:
    device = torch.device("cpu") 
    net = Net().to(device)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    client = clients[randint(0,2)]
    return FlowerNumPyClient(partition_id, net, trainloader, valloader, client["cpu_time"]).to_client()

def create_client_app(clients):
# Create the ClientApp
    return ClientApp(client_fn=lambda context: numpyclient_fn(context, clients))
