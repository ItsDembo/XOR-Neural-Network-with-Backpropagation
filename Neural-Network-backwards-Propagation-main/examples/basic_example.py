import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xor_nn.model import XORNeuralNetwork
import numpy as np

# Configuration - Easy to modify!
CONFIG = {
    'hidden_size': 2,
    'learning_rate': 0.1,
    'epochs': 100000,
    'random_seed': 42
}

def main():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([0, 1, 1, 0])
    
    # Initialize and train
    nn = XORNeuralNetwork(
        hidden_size=CONFIG['hidden_size'],
        random_seed=CONFIG['random_seed']
    )
    
    print("Training network...")
    errors = nn.train(X, y, 
                    lr=CONFIG['learning_rate'],
                    epochs=CONFIG['epochs'])
    
    # Test predictions
    _, predictions = nn.forward(X)
    print("\nPredictions vs Ground Truth:")
    for i in range(X.shape[1]):
        print(f"Input: {X[:,i]}, Prediction: {predictions[0,i]:.4f}, Target: {y[i]}")
    
    print(f"\nFinal MAE: {np.mean(np.abs(y - predictions)):.4f}")

if __name__ == "__main__":
    main()