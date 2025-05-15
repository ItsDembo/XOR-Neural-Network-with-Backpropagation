import numpy as np
from typing import Tuple

class XORNeuralNetwork:
    """Core 2-layer neural network for XOR problem"""
    
    def __init__(self, hidden_size: int = 2, random_seed: int = None):
        """
        Initialize network with given hidden layer size
        
        Args:
            hidden_size: Number of neurons in hidden layer
            random_seed: Optional seed for reproducibility
        """
        if random_seed:
            np.random.seed(random_seed)
            
        # Initialize weights and biases
        self.w1 = np.random.randn(hidden_size, 2) * 0.1  # Input to hidden
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(1, hidden_size) * 0.1  # Hidden to output
        self.b2 = np.zeros((1, 1))
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network
        
        Args:
            X: Input data (2xN matrix)
            
        Returns:
            (hidden_layer_output, final_output)
        """
        self.z1 = np.dot(self.w1, X) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a1, self.a2
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             lr: float = 0.1, epochs: int = 100000) -> np.ndarray:
        """
        Train network using backpropagation
        
        Args:
            X: Training inputs (2xN)
            y: Training labels (1xN)
            lr: Learning rate
            epochs: Number of iterations
            
        Returns:
            Array of error values
        """
        y = y.reshape(1, -1)  # Ensure proper shape
        errors = np.zeros(epochs//1000 + 1)
        
        for epoch in range(epochs):
            _, a2 = self.forward(X)
            error = y - a2
            
            # Backpropagation
            da2 = error * (a2 * (1 - a2))
            self.w2 += lr * np.dot(da2, self.a1.T)
            self.b2 += lr * np.sum(da2, axis=1, keepdims=True)
            
            da1 = np.dot(self.w2.T, da2)
            dz1 = da1 * (self.a1 * (1 - self.a1))
            self.w1 += lr * np.dot(dz1, X.T)
            self.b1 += lr * np.sum(dz1, axis=1, keepdims=True)
            
            if epoch % 1000 == 0:
                errors[epoch//1000] = np.mean(np.abs(error))
                
        return errors