"""
Deep Q-Network (DQN) Agent pour Trading
========================================

Implémentation d'un agent DQN pour le trading automatisé avec visualisation en temps réel.

Fonctionnalités:
- Réseau de neurones avec replay buffer
- Exploration vs exploitation (epsilon-greedy)
- Visualisation des performances en temps réel
- Sauvegarde des modèles et métriques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, namedtuple
import os
import pickle
from datetime import datetime
import pandas as pd

# Transition pour le replay buffer
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Réseau de neurones pour l'agent DQN
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=4, dropout=0.2):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Couches cachées
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Buffer de replay pour stocker les expériences
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajouter une transition au buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonner un batch de transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agent DQN pour le trading
    """
    
    def __init__(self, 
                 state_size,
                 action_size=4,
                 lr=0.001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=32,
                 target_update=100,
                 device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device (GPU si disponible)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent DQN utilise: {self.device}")
        
        # Réseaux de neurones
        self.q_network = DQNNetwork(state_size, output_size=action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, output_size=action_size).to(self.device)
        
        # Copier les poids initiaux
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Compteurs
        self.steps = 0
        self.episode = 0
        
        # Métriques pour visualisation
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'epsilon': [],
            'loss': [],
            'net_worth': [],
            'actions': [],
            'q_values': []
        }
    
    def act(self, state, training=True):
        """
        Choisir une action selon la politique epsilon-greedy
        """
        if training and random.random() < self.epsilon:
            # Exploration aléatoire
            action = random.randrange(self.action_size)
            q_values = [0.0] * self.action_size
        else:
            # Exploitation: choisir la meilleure action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values_tensor = self.q_network(state_tensor)
                q_values = q_values_tensor.cpu().numpy()[0]
                action = np.argmax(q_values)
        
        return action, q_values
    
    def remember(self, state, action, reward, next_state, done):
        """Stocker une expérience dans le replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Entraîner le réseau sur un batch d'expériences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Échantillonner un batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convertir en tenseurs
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Q-values actuelles
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Q-values futures (target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Mise à jour epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Mise à jour du target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_history(self, episode_reward, net_worth, action, q_values, loss=None):
        """Mettre à jour l'historique pour la visualisation"""
        self.training_history['episodes'].append(self.episode)
        self.training_history['rewards'].append(episode_reward)
        self.training_history['epsilon'].append(self.epsilon)
        self.training_history['net_worth'].append(net_worth)
        self.training_history['actions'].append(action)
        self.training_history['q_values'].append(q_values)
        
        if loss is not None:
            self.training_history['loss'].append(loss)
        
        self.episode += 1
    
    def save_model(self, filepath):
        """Sauvegarder le modèle et l'historique"""
        checkpoint = {
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            },
            'steps': self.steps,
            'episode': self.episode
        }
        
        torch.save(checkpoint, filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath):
        """Charger un modèle sauvegardé"""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
                
                self.q_network.load_state_dict(checkpoint['model_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_history = checkpoint['training_history']
                self.steps = checkpoint['steps']
                self.episode = checkpoint['episode']
                
                # Restaurer l'epsilon
                self.epsilon = checkpoint['hyperparameters']['epsilon']
                
                print(f"Modèle chargé: {filepath}")
                print(f"Episode: {self.episode}, Steps: {self.steps}, Epsilon: {self.epsilon:.4f}")
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
                print("Démarrage avec un nouveau modèle...")
                return False
        else:
            print(f"Fichier de modèle non trouvé: {filepath}")
            return False


class TradingVisualizer:
    """
    Visualiseur en temps réel pour l'entraînement DQN
    """
    
    def __init__(self, agent, update_interval=100):
        self.agent = agent
        self.update_interval = update_interval
        
        # Configuration matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('DQN Trading Agent - Entraînement en Temps Réel', fontsize=16, color='white')
        
        # Initialisation des graphiques
        self.setup_plots()
        
        # Animation
        self.ani = None
    
    def setup_plots(self):
        """Configurer les sous-graphiques"""
        # Récompenses par épisode
        self.axes[0, 0].set_title('Récompenses par Episode', color='white')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Récompense')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Net Worth
        self.axes[0, 1].set_title('Net Worth Evolution', color='white')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Net Worth ($)')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon
        self.axes[0, 2].set_title('Epsilon (Exploration)', color='white')
        self.axes[0, 2].set_xlabel('Episode')
        self.axes[0, 2].set_ylabel('Epsilon')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        # Loss
        self.axes[1, 0].set_title('Training Loss', color='white')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution des actions
        self.axes[1, 1].set_title('Distribution des Actions', color='white')
        self.action_names = ['HOLD', 'BUY', 'SELL', 'SHORT']
        
        # Q-values moyennes
        self.axes[1, 2].set_title('Q-Values Moyennes', color='white')
        self.axes[1, 2].set_xlabel('Episode')
        self.axes[1, 2].set_ylabel('Q-Value')
        self.axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update_plots(self, frame):
        """Mettre à jour les graphiques"""
        history = self.agent.training_history
        
        if len(history['episodes']) == 0:
            return
        
        # Nettoyer les axes
        for ax in self.axes.flat:
            ax.clear()
        
        self.setup_plots()
        
        episodes = history['episodes']
        
        # Récompenses
        if history['rewards']:
            self.axes[0, 0].plot(episodes, history['rewards'], 'cyan', alpha=0.7)
            if len(history['rewards']) > 10:
                # Moyenne mobile
                window = min(10, len(history['rewards']))
                rewards_ma = pd.Series(history['rewards']).rolling(window).mean()
                self.axes[0, 0].plot(episodes, rewards_ma, 'yellow', linewidth=2, label=f'MA({window})')
                self.axes[0, 0].legend()
        
        # Net Worth
        if history['net_worth']:
            self.axes[0, 1].plot(episodes, history['net_worth'], 'green', alpha=0.7)
            initial_balance = 10000  # Ajuster selon votre config
            self.axes[0, 1].axhline(y=initial_balance, color='white', linestyle='--', alpha=0.5, label='Initial')
            self.axes[0, 1].legend()
        
        # Epsilon
        if history['epsilon']:
            self.axes[0, 2].plot(episodes, history['epsilon'], 'orange', alpha=0.7)
        
        # Loss
        if history['loss']:
            self.axes[1, 0].plot(episodes[-len(history['loss']):], history['loss'], 'red', alpha=0.7)
        
        # Distribution des actions
        if history['actions']:
            action_counts = [history['actions'].count(i) for i in range(4)]
            colors = ['blue', 'green', 'red', 'purple']
            self.axes[1, 1].bar(self.action_names, action_counts, color=colors, alpha=0.7)
            self.axes[1, 1].set_ylabel('Nombre d\'actions')
        
        # Q-values moyennes
        if history['q_values']:
            q_means = [np.mean(q_vals) if len(q_vals) > 0 else 0 for q_vals in history['q_values']]
            self.axes[1, 2].plot(episodes, q_means, 'magenta', alpha=0.7)
        
        # Informations en temps réel
        if episodes:
            last_episode = episodes[-1]
            last_reward = history['rewards'][-1] if history['rewards'] else 0
            last_net_worth = history['net_worth'][-1] if history['net_worth'] else 0
            current_epsilon = history['epsilon'][-1] if history['epsilon'] else 0
            
            info_text = f"Episode: {last_episode} | Reward: {last_reward:.4f} | Net Worth: ${last_net_worth:.2f} | ε: {current_epsilon:.4f}"
            self.fig.suptitle(f'DQN Trading Agent - {info_text}', fontsize=12, color='white')
    
    def start_animation(self):
        """Démarrer l'animation en temps réel"""
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, interval=self.update_interval, blit=False
        )
        plt.show(block=False)
    
    def stop_animation(self):
        """Arrêter l'animation"""
        if self.ani:
            self.ani.event_source.stop()
    
    def save_plots(self, filepath):
        """Sauvegarder les graphiques"""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Graphiques sauvegardés: {filepath}")


def get_action_name(action):
    """Convertir le numéro d'action en nom"""
    action_names = ['HOLD', 'BUY', 'SELL', 'SHORT']
    return action_names[action] if 0 <= action < len(action_names) else 'UNKNOWN'


if __name__ == "__main__":
    print("Classe DQN prête!")
    print("Utilisez 'python dqn_trader.py' pour lancer l'entraînement")
    
    # Test basique de la classe
    print("\nTest de la classe DQN...")
    
    # Créer un agent test
    agent = DQNAgent(state_size=13, action_size=4)
    print(f"Agent créé avec {agent.state_size} inputs et {agent.action_size} actions")
    
    # Test d'action
    test_state = np.random.random(13)
    action, q_values = agent.act(test_state)
    print(f"Action test: {get_action_name(action)} (Q-values: {q_values})")
    
    print("Classe DQN fonctionnelle!")