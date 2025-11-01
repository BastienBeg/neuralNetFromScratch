# Création du data loader pour le dataset Breast Cancer de scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from config import Config

class DataLoader:
    def __init__(self, test_size=0.2, random_state=Config.random_seed):
        data = load_breast_cancer()
        X = data.data
        Y = data.target.reshape(-1, 1)  # Reshape pour avoir une colonne
        
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
    
    def get_train_data(self):
        return self.X_train, self.Y_train
    
    def get_val_data(self):
        return self.X_val, self.Y_val