class Config:
    learning_rate = 0.01
    epochs = 1000
    batch_size = 32
    input_size = 30  # Nombre de features dans le dataset Breast Cancer
    output_size = 1  # Classification binaire
    hidden_layers = [64, 32]  # Exemple de couches cachées
    random_seed = 42