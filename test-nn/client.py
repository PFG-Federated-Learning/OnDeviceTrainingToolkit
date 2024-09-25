import torch
from typing import OrderedDict
from flwr.client import Client, ClientApp, NumPyClient
from model import train, test
from transformers import AutoModelForSequenceClassification, AdamW
from dataset import load_data

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader), {"accuracy": accuracy}
    
def client_fn(context: Context):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = 10
    model_name = "prajjwal1/bert-tiny"
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    model_name = "prajjwal1/bert-tiny"
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

# Flower ClientApp
app = ClientApp(client_fn)