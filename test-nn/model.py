from transformers import AutoModelForSequenceClassification, AdamW
import torch

def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader, device):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy