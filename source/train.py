# Creation d'un trainer pour un reseau de neurones simple
from neural_net import NeuralNet
from config import Config
from data_loader import DataLoader
import numpy as np
class trainer:
    def __init__(self, config: Config, data_loader: DataLoader, model : NeuralNet):
        self.config = config
        self.data_loader = data_loader
        self.model = model
    
    def train(self):
        X_train, Y_train = self.data_loader.get_train_data()
        X_val, Y_val = self.data_loader.get_val_data()
        
        for epoch in range(self.config.epochs):
            # Forward pass
            Y_hat = self.model.forward(X_train)
            loss = self.model.compute_loss(Y_hat, Y_train)
            # Backward pass
            self.model.backward(Y_hat, Y_train)
            # Mise à jour des paramètres
            self.model.update_parameters(self.config.learning_rate)
            if epoch % 40 == 0:
                val_Y_hat = self.model.forward(X_val)
                val_loss = self.model.compute_loss(val_Y_hat, Y_val)
                evaluate = self.evaluate()
                print(f"Epoch {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")
    
    def evaluate(self):
        X_val, Y_val = self.data_loader.get_val_data()
        Y_pred = self.model.predict(X_val)
        accuracy = np.mean(Y_pred == Y_val) * 100
        print(f"Validation Accuracy: {accuracy}%")