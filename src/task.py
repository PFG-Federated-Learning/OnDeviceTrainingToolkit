"""test-transformers: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


fds = None  # Cache FederatedDataset

# Load and preprocess the dataset
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_set, test_set = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, test_loader

 
def load_data(partition_id: int, num_partitions: int, model_name: str, dataset: str):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
        dataset="stanfordnlp/imdb",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(model, train_loader):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1):  # Single epoch for demonstration
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)