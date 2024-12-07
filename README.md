# On-device Training - Model & Dataset Generation and Federated simulation

This repository aims to provide all of the supporting features for **On-Device Training** android application, that can be seen in this [github repository](https://github.com/PFG-Federated-Learning/OnDeviceTraning).

The repository cointains the features for creating datasets and model binary files to be used for training, as well as a simulation environment for the device metrics collected by the app. Have a look at more descriptive information for each of the 2 sections below:

- [Model and Dataset Generation](/model_generation/README)
- [Federated Simulation](/flower_simulation/README)

## Repository Structure

```
.
├── deviceConfigurations.json
├── .gitignore
├── README
├── model_generation
│   ├── example_cifar10
│   │   ├── constants.py
│   │   ├── dataset_definition.py
│   │   ├── main.py
│   │   └── model_definition.py
|   ├── example_mnist
│   │   ├── constants.py
│   │   ├── dataset_definition.py
│   │   ├── main.py
│   │   └── model_definition.py
│   └── README
├── requirements.txt
└── simulation
    ├── client_app.py
    ├── config.yaml
    ├── firebase_retriever.py
    ├── README
    ├── server_app.py
    ├── simulation.py
    └── task.py
```
