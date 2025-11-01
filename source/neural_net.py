import numpy as np
from config import Config
np.random.seed(Config.random_seed)  # Pour la reproductibilité

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
        Effectue le feed forward à travers le réseau.
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
    
    def compute_loss(self, Y_hat, Y):
        """
        Calcule la loss binaire cross-entropy avec l'ajout de 1e-8 pour éviter le log(0).
        """
        m = Y.shape[0]
        loss = - (1/m) * np.sum(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8))
        return loss
    
    def backward(self, Y_hat, Y):
        """
        Effectue la rétropropagation pour calculer les gradients.
        """
        m = Y.shape[0]
        dA = - (np.divide(Y, Y_hat + 1e-8) - np.divide(1 - Y, 1 - Y_hat + 1e-8))
        
        for i in reversed(range(1, len(self.layer_sizes))):
            dZ = dA * (Y_hat * (1 - Y_hat)) if i == len(self.layer_sizes) - 1 else dA * (self.caches[f"Z{i}"] > 0)
            A_prev = self.caches[f"A{i-1}"]
            self.gradients[f"dW{i}"] = (1/m) * np.dot(A_prev.T, dZ)
            self.gradients[f"db{i}"] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, self.weights[f"w{i}"].T)
    
    def update_parameters(self, learning_rate):
        """
        Met à jour les poids et les biais en utilisant les gradients calculés.
        """
        for i in range(1, len(self.layer_sizes)):
            self.weights[f"w{i}"] -= learning_rate * self.gradients[f"dW{i}"]
            self.biases[f"b{i}"] -= learning_rate * self.gradients[f"db{i}"]
    
    def predict(self, X):
        """
        Fait des prédictions binaires basées sur une threshold de 0.5.
        """
        Y_hat = self.forward(X)
        predictions = (Y_hat > 0.5).astype(int)
        return predictions
    
    def save_model(self, filepath):
        """
        Sauvegarde les poids et biais du modèle dans un fichier.
        """
        np.savez(filepath, weights=self.weights, biases=self.biases)