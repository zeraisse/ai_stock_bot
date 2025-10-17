import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import DQNModel as model

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Dimensions de l'état (taille de l'observation) et du nombre d'actions
        self.state_size = state_size
        self.action_size = action_size
        # Mémoire de rejouage pour stocker les transitions (s, a, r, s', done)
        self.memory = deque(maxlen=2000)
        # Facteur d'actualisation (discount) pour les récompenses futures
        self.gamma = 0.95
        # Paramètres epsilon-greedy pour l'exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Taux d'apprentissage du réseau
        self.learning_rate = 0.001

        # Utilise CUDA si disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseau Q (approximation des Q(s,a))
        self.model = model.DQNModel(state_size, action_size).to(self.device)
        # Optimiseur Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Fonction de perte (MSE entre Q prédits et cibles)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        # Politique epsilon-greedy: exploration avec proba epsilon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Exploitation: choisir l'action avec Q maximal
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Stocke une transition dans la mémoire
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Lance l'apprentissage uniquement si assez d'échantillons
        if len(self.memory) < batch_size:
            return

        # Échantillonnage aléatoire d'un mini-lot
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Conversion des états en tenseurs
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            # Q(s, ·) prédits par le réseau courant
            q_values = self.model(state)

            with torch.no_grad():
                # Q(s', ·) pour l'état suivant (max_a' Q(s', a'))
                next_q = self.model(next_state)
                target = reward if done else reward + self.gamma * torch.max(next_q).item()

            # Cible: Q(s, a) doit approcher 'target'; clone pour ne pas backprop sur la cible
            target_f = q_values.clone().detach()
            target_f[0][action] = target

            # Optimisation: MSE(Q(s, ·), cible) puis mise à jour des poids
            self.optimizer.zero_grad()
            loss = self.loss_fn(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        # Décroissance de l'exploration epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        # Sauvegarde des poids du modèle
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # Chargement des poids du modèle
        self.model.load_state_dict(torch.load(path))
