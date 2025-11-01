# neuralNetFromScratch

Ce projet a un but purement pédagogique. Il consiste à implémenter un réseau de neurones artificiels à partir de zéro en Python, sans utiliser de bibliothèques spécialisées comme TensorFlow ou PyTorch. L'objectif est de comprendre les concepts fondamentaux derrière les réseaux de neurones, y compris la propagation avant, la rétropropagation et l'optimisation.

## Structure du projet

On retrouve les fichiers suivants dans ce projet :
- `neural_net.py` : Contient l'implémentation principale du réseau de neurones, y compris les classes et les fonctions pour la création, l'entraînement et l'évaluation du modèle.
- `data_loader.py` : Fournit des fonctions pour charger et prétraiter les données utilisées pour entraîner le réseau de neurones.
- `train.py` : Script principal pour entraîner le réseau de neurones en utilisant les données chargées.
- `utils.py` : Contient des fonctions utilitaires pour diverses opérations, telles que le calcul de la perte et des métriques d'évaluation.
- `main.py` : Point d'entrée du programme qui orchestre le chargement des données, l'entraînement du modèle et l'évaluation des performances.
- `explore_data.ipynb` : Notebook Jupyter pour explorer et visualiser les données utilisées dans le projet.
- `README.md` : Ce fichier, qui fournit une vue d'ensemble du projet.
- `config.py` : Fichier de configuration pour définir les hyperparamètres du modèle et les paramètres d'entraînement.

## Datasets utilisé 

Le projet utilise le dataset Breast cancer disponible sur sklearn pour entraîner et évaluer le réseau de neurones. Dont les caractéristiques sont les suivantes :
- Nombre d'échantillons : 569
- Nombre de caractéristiques : 30
- Nombre de classes : 2 (malin et bénin)

## Fonctionnement détaillé du réseau de neurones

Le réseau de neurones implémenté dans ce projet suit une architecture simple avec une couche d'entrée, une ou plusieurs couches cachées et une couche de sortie. Voici un aperçu du fonctionnement détaillé :

### Definition de l'architecture 

L'utilisateur peut spécifier le nombre de couches cachées, le nombre de neurones par couche, et les fonctions d'activation utilisées.

Dans ce projet, nous utilisons une architecture avec :
- Couche d'entrée : 30 neurones (correspondant aux 30 caractéristiques du dataset)
- Couche cachée 1 : 64 neurones, fonction d'activation ReLU
- Couche cachée 2 : 32 neurones, fonction d'activation ReLU
- Couche de sortie : 1 neurone, fonction d'activation sigmoïde (pour la classification binaire)

### Propagation avant

Les données d'entrée sont propagées à travers le réseau couche par couche, en appliquant les poids et les biais, ainsi que les fonctions d'activation pour produire une sortie.

Dans notre implémentation, la propagation avant se fait de la manière suivante :
- Notre entrée X (de dimension 30) est multipliée par les poids W1 de la première couche cachée (de dimension 30x64), puis on ajoute le biais b1 (de dimension 64) et on applique la fonction d'activation ReLU pour obtenir la sortie de la première couche cachée A1 (de dimension 64).
- La sortie A1 est ensuite multipliée par les poids W2 de la deuxième couche cachée (de dimension 64x32), on ajoute le biais b2 (de dimension 32) et on applique à nouveau la fonction d'activation ReLU pour obtenir la sortie de la deuxième couche cachée A2 (de dimension 32).
- Enfin, la sortie A2 est multipliée par les poids W3 de la couche de sortie (de dimension 32x1), on ajoute le biais b3 (de dimension 1) et on applique la fonction d'activation sigmoïde pour obtenir la sortie finale Y_hat (de dimension 1), qui représente la probabilité que l'entrée appartienne à la classe positive (malin).

Z1 = X.dot(W1) + b1
A1 = ReLU(Z1)
Z2 = A1.dot(W2) + b2
A2 = ReLU(Z2)
Z3 = A2.dot(W3) + b3
Y_hat = sigmoid(Z3)

Avec :
- Relu(x) = max(0, x)
- Sigmoid(x) = 1 / (1 + exp(-x))

Nous avons donc un total de 30*64 + 64*32 + 32*1 = 1920 + 2048 + 32 = 4000 poids, plus les biais associés.

### Calcul de la perte

La sortie du réseau est comparée aux étiquettes réelles pour calculer la perte à l'aide d'une fonction de perte appropriée.

Dans ce projet, nous utilisons la fonction de perte binaire cross-entropy pour mesurer la différence entre les prédictions Y_hat et les étiquettes réelles Y. La perte est calculée comme suit :
- Loss = - (1/m) * Σ [ Y * log(Y_hat) + (1 - Y) * log(1 - Y_hat) ]
où m est le nombre d'échantillons dans le batch.

On voit ici l'importance de la fonction sigmoïde en sortie, car elle garantit que les prédictions Y_hat sont comprises entre 0 et 1, ce qui est nécessaire pour le calcul de la cross-entropy. Plus de details sur le fonctionnement de la foncition sur le lien suivant : https://www.datacamp.com/fr/tutorial/the-cross-entropy-loss-function-in-machine-learning

### Rétropropagation

Les gradients de la perte par rapport aux poids et aux biais sont calculés en utilisant la rétropropagation, permettant ainsi de mettre à jour les paramètres du modèle.

Dans notre implémentation, la rétropropagation se fait en calculant les dérivées partielles de la perte par rapport aux sorties de chaque couche, puis en utilisant la règle de la chaîne pour propager ces gradients en arrière à travers le réseau. Les gradients sont calculés comme suit :

Sachant que l'on cherche à minimiser la loss L, il nous faut calculer les dérivées partielles suivantes :
- dL/dW3, dL/db3
- dL/dW2, dL/db2
- dL/dW1, dL/db1

les calculs se font en utilisant les sorties intermédiaires A1, A2 et les dérivées des fonctions d'activation ReLU et sigmoïde.

- dL/dY_hat = -1/m * (Y / Y_hat - (1 - Y) / (1 - Y_hat))
- dY_hat/dZ3 = Y_hat * (1 - Y_hat)  (dérivée de la sigmoïde)
- dZ3/dW3 = A2
- dZ3/db3 = 1
Soit :
- dL/dW3 = dL/dY_hat * dY_hat/dZ3 * dZ3/dW3 = -1/m * (Y / Y_hat - (1 - Y) / (1 - Y_hat)) * Y_hat * (1 - Y_hat) * A2 
- dL/db3 = dL/dY_hat * dY_hat/dZ3 * dZ3/db3 = -1/m * (Y / Y_hat - (1 - Y) / (1 - Y_hat)) * Y_hat * (1 - Y_hat) * 1

En simplifiant, on obtient :
- dL/dW3 = (1/m) * A2.T.dot(Y_hat - Y)
- dL/db3 = (1/m) * Σ (Y_hat - Y)

De manière similaire, on calcule les gradients pour les couches cachées en utilisant les dérivées des fonctions ReLU. Puis on obtient les formules suivantes :

#### couche sortie
dZ3 = Y_hat - Y                     # (m,1)
dW3 = (A2.T @ dZ3) / m              # (32,1)
db3 = np.sum(dZ3, axis=0) / m       # (1,)

#### couche 2 (ReLU)
dA2 = dZ3 @ W3.T                    # (m,32)
dZ2 = dA2 * (Z2 > 0)                # (m,32)   element-wise
dW2 = (A1.T @ dZ2) / m              # (64,32)
db2 = np.sum(dZ2, axis=0) / m       # (32,)

#### couche 1 (ReLU)
dA1 = dZ2 @ W2.T                    # (m,64)
dZ1 = dA1 * (Z1 > 0)                # (m,64)
dW1 = (X.T @ dZ1) / m               # (30,64)
db1 = np.sum(dZ1, axis=0) / m       # (64,)


## Optimisation 

On utilise la descente de gradient pour mettre à jour les poids et les biais en fonction des gradients calculés lors de la rétropropagation.
On met à jour les poids et les biais comme suit :
- W = W - learning_rate * dW
- b = b - learning_rate * db



## Évaluation

Après l'entraînement, le modèle est évalué sur un ensemble de test pour mesurer ses performances en termes de précision, rappel, F1-score, etc.

