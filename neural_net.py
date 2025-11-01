import numpy as np
from config import Config
np.random.seed(42)  # Pour la reproductibilité

class NeuralNet:
    def __init__(self, config : Config):
        """
        Prend une instance de la classe Config en paramètre pour initialiser les hyperparamètres du réseau.
        """
        self.weights = {}
        self.biases = {}
        self.caches = {}
        self.gradients = {}
        self.layer_sizes = [config.input_size] + config.hidden_layers + [config.output_size]
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialise les poids et les biais du réseau de manière aléatoire.
        """
        for i in range(1, len(self.layer_sizes)):
            self.weights[f"w{i}"] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.01
            self.biases[f"b{i}"] = np.zeros((1, self.layer_sizes[i]))
    
    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        """
        Effectue la propagation avant à travers le réseau.
        """
        for i in range(1, len(self.layer_sizes)):
            Z = np.dot(X, self.weights[f"w{i}"]) + self.biases[f"b{i}"]
            if i == len(self.layer_sizes) - 1:
                A = self._sigmoid(Z)
            else:
                A = self._relu(Z)
            self.caches[f"A{i-1}"] = X
            self.caches[f"Z{i}"] = Z
            X = A
        return A