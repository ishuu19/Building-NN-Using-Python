# Neural Network From Scratch Using Python

A neural network built from scratch using **Python and NumPy**, created to understand what is actually happening inside a neural network instead of relying on high-level frameworks such as TensorFlow or PyTorch.

The project starts with the basic mathematical building blocks of a neural network — dense layers, activation functions, forward propagation, loss functions, backpropagation, and optimizers — and is designed to gradually grow toward training on real datasets and images.

## Why I Built This

Neural-network libraries make it very easy to create a model with only a few lines of code, but they can hide much of the underlying mathematics.

The goal of this project is to build those components manually and understand:

- how inputs move through a network
- how weights and biases create predictions
- why activation functions are needed
- how loss measures prediction error
- how gradients are calculated during backpropagation
- how optimizers update parameters
- how the same numerical layers can eventually be used for image data

The focus is on **learning by building**.

## Current Features

### Layers

- `Dense`
  - Weight and bias initialization
  - Forward propagation
  - Backpropagation
  - Batch-based gradient calculations

### Activation Functions

- `ReLU`
- `LeakyReLU`
- `Sigmoid`

Each activation supports both forward and backward propagation.

### Network

- `NeuralNetwork`
  - Add layers sequentially
  - Forward propagation through all layers
  - Backpropagation through layers in reverse order

### Loss Functions

- `MSELoss`
  - Mean Squared Error
  - Forward and backward pass

- `CrossEntropyLoss`
  - Softmax probabilities
  - Numerical-stability adjustment
  - Multiclass cross-entropy
  - Backward gradient calculation

### Optimizers

The optimizer module currently contains implementations of:

- SGD
- Momentum
- Nesterov Momentum
- RMSProp
- Adam

## Project Structure

```text
Building-NN-Using-Python/
│
├── NeuralNetwork.py
│   ├── Dense
│   ├── Sigmoid
│   ├── ReLU
│   ├── LeakyReLU
│   ├── NeuralNetwork
│   ├── MSELoss
│   └── CrossEntropyLoss
│
├── Optimizers.py
│   ├── SGD
│   ├── Momentum
│   ├── Nesterov
│   ├── RMSProp
│   └── Adam
│
├── Neural Network with Python from scratch.ipynb
│
└── README.md
```

## How the Network Works

The basic training process is:

```text
Input
  ↓
Dense Layer
  ↓
Activation
  ↓
Dense Layer
  ↓
Prediction
  ↓
Loss
  ↓
Backward Propagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Weights
```

This process repeats over many training iterations until the model improves its predictions.

## Example Architecture

A network can be constructed by combining reusable layers:

```python
network = NeuralNetwork()

network.add(Dense(3, 5))
network.add(ReLU())
network.add(Dense(5, 2))
```

Conceptually:

```text
3 input features
       ↓
Dense: 3 → 5
       ↓
ReLU
       ↓
Dense: 5 → 2
       ↓
2 outputs
```

The important idea is that the layers only work with numbers.

That means the same network can later accept image data once an image has been converted into numerical pixel values.

## Batch-Based Inputs

The current dense-layer implementation is designed around **2D batches**.

For example, a single sample with three features should be represented as:

```text
shape = (1, 3)
```

rather than:

```text
shape = (3,)
```

Multiple samples naturally become:

```text
(batch_size, number_of_features)
```

For example:

```text
100 samples × 3 features
→ shape (100, 3)
```

This same structure will later allow flattened images to be passed into the network:

```text
600 images × 784 pixels
→ shape (600, 784)
```

## Images Later

An image is ultimately a collection of numerical pixel values.

For example, an MNIST image has:

```text
28 × 28 = 784 pixels
```

A future image pipeline can therefore be:

```text
Image
  ↓
Load image
  ↓
Convert to NumPy array
  ↓
Normalize pixels
  ↓
Flatten
  ↓
784 numerical features
  ↓
Neural Network
```

The goal is to keep image processing separate from the neural-network layers so that the same core network can work with different types of numerical data.

## Installation

Clone the repository:

```bash
git clone https://github.com/ishuu19/Building-NN-Using-Python.git
cd Building-NN-Using-Python
```

Install NumPy:

```bash
pip install numpy
```

To run the notebook, Jupyter can also be installed:

```bash
pip install jupyter
```

Then launch:

```bash
jupyter notebook
```

## Requirements

- Python 3
- NumPy
- Jupyter Notebook (optional, for running the notebook)

No deep-learning framework is required.

## Roadmap

Planned additions include:

- mini-batch training utilities
- improved weight initialization
- train / validation / test workflows
- accuracy metrics
- model saving and loading
- image preprocessing
- MNIST handwritten-digit classification
- dropout
- regularization
- convolutional layers
- pooling layers
- CNN support
- automatic differentiation

The long-term goal is to gradually turn the project into a small educational deep-learning framework.

## Learning Goal

This repository is primarily an educational project.

Instead of only learning how to *use* a neural network library, the aim is to understand how the important pieces work internally:

```text
Forward Pass
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
Optimization
      ↓
Learning
```

## Disclaimer

This project is being built for learning and experimentation rather than production use. Components are intentionally implemented manually so the underlying mathematics and data flow remain visible.

## Author

Built by **ishuu19** while learning neural networks from the ground up.
