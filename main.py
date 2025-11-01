from neural_net import NeuralNet
from config import Config
import numpy as np

if __name__ == "__main__":
    config = Config()
    model = NeuralNet(config)
    print("Neural Network initialized with the following layer sizes:")
    print(model.weights.keys())
    for key in model.weights:
        print(f"{key}: {model.weights[key].shape}")
    print(model.biases.keys())
    for key in model.biases:
        print(f"{key}: {model.biases[key].shape}")
    
    # Exemple de propagation avant avec des données aléatoires
    X_dummy = np.random.randn(5, config.input_size)  # 5 exemples, input_size features
    output = model.forward(X_dummy)
    print("Output of the forward pass : \n", output)