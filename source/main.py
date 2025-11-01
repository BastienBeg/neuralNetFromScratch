from neural_net import NeuralNet
from config import Config
from train import trainer
from data_loader import DataLoader

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
    
    data_loader = DataLoader()
    model = NeuralNet(config)
    trainer_instance = trainer(config, data_loader, model)
    trainer_instance.train()
    model.save_model("breast_cancer_model.npz")

