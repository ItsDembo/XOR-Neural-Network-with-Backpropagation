# XOR Neural Network with Backpropagation

![XOR Decision Boundary](examples/xor_results.png)

A Python implementation of a 2-layer neural network that learns the XOR function through backpropagation, complete with training visualization and decision boundary plotting.

## Features

- **Core Implementation**:
  - Customizable hidden layer size
  - Sigmoid activation functions
  - Backpropagation learning
  - Weight initialization with seed control

- **Visualization Tools**:
  - Training error over time
  - Prediction vs target comparison
  - Decision boundary plotting
  - High-resolution image export

- **Experimentation Ready**:
  - Easy parameter adjustment
  - Reproducible results (seed control)
  - Clean separation of core and example code

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Neural-Network-backwards-Propagation.git
cd Neural-Network-backwards-Propagation
Install dependencies:

bash
pip install -r requirements.txt
Project Structure
.
├── xor_nn/                   # Core implementation
│   ├── __init__.py
│   └── model.py              # XORNeuralNetwork class
│
├── examples/                 # Usage examples
│   ├── basic_example.py      # Simple text-based demo
│   └── visualization.py      # Graphical analysis
│
├── requirements.txt          # Dependencies
└── README.md                 # This file
Usage
Basic Example
python
from xor_nn.model import XORNeuralNetwork
import numpy as np

# Initialize network
nn = XORNeuralNetwork(hidden_size=2, random_seed=42)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
y = np.array([0, 1, 1, 0])

# Train and evaluate
nn.train(X, y, lr=0.1, epochs=100000)
predictions = nn.forward(X)[1]
Running Examples
Basic text output:

bash
python -m examples.basic_example
Visual analysis:

bash
python -m examples.visualization
This will generate:

training_error.png: Learning curve

xor_results.png: Predictions and decision boundary

Configuration Options
Parameter	Default	Description
hidden_size	2	Neurons in hidden layer
random_seed	None	Seed for reproducible results
lr	0.1	Learning rate
epochs	100,000	Training iterations
Results Example
Input [0 0]: Pred 0.0123 (Target 0)
Input [0 1]: Pred 0.9876 (Target 1)
Input [1 0]: Pred 0.9872 (Target 1)
Input [1 1]: Pred 0.0128 (Target 0)
Final MAE: 0.0010
Training Error

Requirements
Python 3.8+

NumPy

Matplotlib