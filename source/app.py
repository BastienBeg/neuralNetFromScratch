from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_cors import CORS
import json
import time
import threading
import numpy as np

# Importez vos classes
from neural_net import NeuralNet
from config import Config
from data_loader import DataLoader

app = Flask(__name__)
CORS(app)

# Variables globales pour partager l'état
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'train_loss': 0,
    'val_loss': 0,
    'accuracy': 0,
    'status': 'idle'
}

class TrainerWithCallbacks:
    def __init__(self, config: Config, data_loader: DataLoader, model: NeuralNet):
        self.config = config
        self.data_loader = data_loader
        self.model = model
        self.callback = None
    
    def set_callback(self, callback):
        """Définit une fonction callback appelée à chaque epoch"""
        self.callback = callback
    
    def train(self):
        X_train, Y_train = self.data_loader.get_train_data()
        X_val, Y_val = self.data_loader.get_val_data()
        
        for epoch in range(self.config.epochs):
            if not training_state['is_training']:
                break
                
            # Forward pass
            Y_hat = self.model.forward(X_train)
            loss = self.model.compute_loss(Y_hat, Y_train)
            
            # Backward pass
            self.model.backward(Y_hat, Y_train)
            
            # Mise à jour des paramètres
            self.model.update_parameters(self.config.learning_rate)
            
            # Calcul des métriques de validation
            val_Y_hat = self.model.forward(X_val)
            val_loss = self.model.compute_loss(val_Y_hat, Y_val)
            Y_pred = self.model.predict(X_val)
            accuracy = np.mean(Y_pred == Y_val) * 100
            
            # Mise à jour de l'état global
            training_state['current_epoch'] = epoch
            training_state['train_loss'] = float(loss)
            training_state['val_loss'] = float(val_loss)
            training_state['accuracy'] = float(accuracy)
            
            # Appel du callback si défini
            if self.callback:
                self.callback(epoch, loss, val_loss, accuracy)
            
            # Petit délai pour rendre la visualisation visible
            time.sleep(0.05)
        
        training_state['is_training'] = False
        training_state['status'] = 'completed'

# Initialisation du modèle et des données
config = Config()
data_loader = DataLoader()
model = NeuralNet(config)
trainer = TrainerWithCallbacks(config, data_loader, model)

@app.route('/')
def index():
    """Sert la page HTML"""
    return render_template('index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('templates', 'styles.css')

@app.route('/start', methods=['POST'])
def start_training():
    """Démarre l'entraînement dans un thread séparé"""
    if not training_state['is_training']:
        training_state['is_training'] = True
        training_state['status'] = 'training'
        training_state['current_epoch'] = 0
        
        # Démarre l'entraînement dans un thread
        thread = threading.Thread(target=trainer.train)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_training'})

@app.route('/stop', methods=['POST'])
def stop_training():
    """Arrête l'entraînement"""
    training_state['is_training'] = False
    training_state['status'] = 'stopped'
    return jsonify({'status': 'stopped'})

@app.route('/reset', methods=['POST'])
def reset_training():
    """Réinitialise le modèle et l'entraînement"""
    global model, trainer
    
    training_state['is_training'] = False
    training_state['current_epoch'] = 0
    training_state['train_loss'] = 0
    training_state['val_loss'] = 0
    training_state['accuracy'] = 0
    training_state['status'] = 'idle'
    
    # Réinitialise le modèle
    model = NeuralNet(config)
    trainer = TrainerWithCallbacks(config, data_loader, model)
    
    return jsonify({'status': 'reset'})

@app.route('/stream')
def stream():
    """Stream SSE des métriques d'entraînement"""
    def generate():
        while True:
            # Envoie les données actuelles
            data = json.dumps(training_state)
            yield f"data: {data}\n\n"
            time.sleep(0.1)  # Mise à jour toutes les 100ms
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/config')
def get_config():
    """Retourne la configuration du modèle"""
    return jsonify({
        'learning_rate': config.learning_rate,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'input_size': config.input_size,
        'output_size': config.output_size,
        'hidden_layers': config.hidden_layers,
        'architecture': [config.input_size] + config.hidden_layers + [config.output_size]
    })

if __name__ == '__main__':
    print("Serveur Flask démarré sur http://localhost:5000")
    print("Ouvrez votre navigateur pour voir la visualisation")
    app.run(debug=True, threaded=True)