from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def load_data(partition_id: int, num_partitions: int, model_name: str):
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
    return trainloader, testloader, data_collator