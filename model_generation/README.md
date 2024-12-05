# Model Generation Examples

This repository provides practical examples of generating models and datasets for on-device training using the **MNIST** and **CIFAR-10** datasets. The goal is to demonstrate how to create TensorFlow models, prepare them for incremental training in TensorFlow Lite (TFLite) format, and serialize datasets into binary format, ready for use in the mobile application.

## Repository Structure

The root folder `model_generation` contains this `README.md` file and two main directories:

- `example_mnist`: Example using the MNIST dataset.
- `example_cifar10`: Example using the CIFAR-10 dataset.

Each of these directories contains four key files:

- **`constants.py`**: Defines constants used in the process, such as hyperparameters and file paths.
- **`dataset_definition.py`**: Contains the function `get_processed_ds()`, which returns the preprocessed training dataset ready for use.
- **`model_definition.py`**: Defines the `MyModel` class, which encapsulates the Keras model backbone and provides functions for training, inference, saving, and restoring the model.
- **`main.py`**: Executes the full flow, including:
  - Model and dataset instantiation.
  - Model training loop.
  - Saving *features* and *labels* in binary format.
  - Converting and saving the Keras model in TFLite format.

## Example Details

### **MNIST Example (`example_mnist`)**
- **Description**: Based on the TensorFlow reference code available [here](https://www.tensorflow.org/datasets/keras_example).
- **Goal**: Demonstrates creating a model for classifying handwritten digits (28x28 pixels, grayscale).
- **Features**:
  - Simple dense network architecture.

### **CIFAR-10 Example (`example_cifar10`)**
- **Description**: Inspired by the GeeksforGeeks article available [here](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/).
- **Goal**: Create a model for classifying color images (32x32 pixels) into 10 categories.
- **Features**:
  - More complex convolutional network architecture.
  
## How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/PFG-Federated-Learning/flower-simulation.git
   cd model_generation
   ```

2. **Run the Desired Example**
   Navigate to the directory of the desired example:
   ```bash
   cd example_mnist  # For the MNIST example
   # or
   cd example_cifar10  # For the CIFAR-10 example
   ```

   Run the `main.py` script:
   ```bash
   python main.py
   ```

3. **Process Outputs**
   - TFLite model saved in the directory specified in `constants.py`.
   - Binary files containing the dataset's *features* and *labels*.

## Code Structure

Below is a summary of the responsibilities of each file:

- **`constants.py`**: Centralizes parameter and path definitions.
- **`dataset_definition.py`**: Handles dataset loading and preprocessing.
- **`model_definition.py`**: Defines the `MyModel` class with:
  - Core functions: `train`, `infer`, `save`, `restore`.
  - Model configuration for incremental training in TFLite.
- **`main.py`**: Integrates the components above to train, save, and export the model.

## Requirements

The scripts use the following packages:

- Python 3.10.12
- TensorFlow 2.13.0
- NumPy 1.24.3
- tqdm 4.66.5

The versions mentioned above are the ones that were tested, but the scripts might work with different versions.

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests.