import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xor_nn.model import XORNeuralNetwork

def plot_network(nn, X, y):
    """Visualize network predictions and decision boundary"""
    _, predictions = nn.forward(X)
    
    plt.figure(figsize=(12, 5))
    
    # Plot predictions vs targets
    plt.subplot(1, 2, 1)
    colors = ['green' if abs(p - t) < 0.1 else 'red' 
              for p, t in zip(predictions[0], y)]
    bars = plt.bar(range(4), predictions[0], color=colors)
    plt.xticks(range(4), [str(x) for x in X.T])
    plt.ylim(0, 1.1)
    plt.title("Predictions vs Targets")
    plt.xlabel("Input")
    plt.ylabel("Output")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    _, Z = nn.forward(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar()
    plt.scatter(X[0], X[1], c=y, s=100, cmap='RdYlBu', edgecolors='black')
    plt.title("Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    
    plt.tight_layout()
    plt.savefig("xor_results.png", dpi=150)
    plt.show()

def main():
    # Create and train network
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([0, 1, 1, 0])
    
    nn = XORNeuralNetwork(hidden_size=2, random_seed=42)
    errors = nn.train(X, y, lr=0.1, epochs=100000)
    
    # Plot training curve (FIXED THE DIMENSION MISMATCH)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(errors)) * 1000, errors)
    plt.title("Training Error Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True)
    plt.savefig("training_error.png", dpi=150)
    plt.show()
    
    # Plot network predictions
    plot_network(nn, X, y)

if __name__ == "__main__":
    main()