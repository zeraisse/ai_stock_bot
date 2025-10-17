"""
Modele DQN avec TFT pour l'extraction de features temporelles.

Je cree ce modele pour combiner:
1. TFT (Temporal Fusion Transformer) - extrait les features des sequences de prix
2. DQN (Deep Q-Network) - apprend la politique de trading optimale

Architecture:
    Input (state) -> TFT (features temporelles) -> FC (Q-values) -> Action
    
Compatible avec DQNAgent et TradingEnv.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tft.tft_predictor import TFTPredictor


class DQNTFTModel(nn.Module):
    """
    Je combine DQN avec TFT pour le trading algorithmique.
    
    TFT extrait des representations riches des sequences temporelles
    (prix, volumes, tendances) puis une couche FC transforme ces features
    en Q-values pour chaque action (HOLD, BUY, SELL).
    """
    
    def __init__(self, state_size, action_size, sequence_length=60):
        """
        J'initialise le modele DQN-TFT.
        
        Args:
            state_size: Taille de l'etat (3 pour balance, holding, price)
            action_size: Nombre d'actions (3 pour HOLD, BUY, SELL)
            sequence_length: Longueur des sequences temporelles (60 jours = ~3 mois de trading)
                            60 jours capture les tendances court/moyen terme sans trop de bruit
        """
        super(DQNTFTModel, self).__init__()

        # Je cree le predicteur TFT avec les memes hyperparametres que LSTM
        # pour assurer une comparaison equitable
        self.tft_predictor = TFTPredictor(
            sequence_length=sequence_length,
            hidden_size=128,  # 128 neurones = bon compromis memoire/capacite pour patterns financiers
            lstm_layers=2,    # 2 couches LSTM dans TFT suffisent pour court et moyen terme
            attention_head_size=4,  # 4 tetes d'attention = 4 perspectives temporelles differentes
            dropout=0.2,      # 20% dropout plus agressif que TFT standard (0.1) car marches tres bruites
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Je construis le modele TFT interne
        # input_size=state_size car je recois (balance, holding, price) du DQNAgent
        self.tft = self.tft_predictor.build_model(input_size=state_size)

        # Couche finale pour calculer les Q-values
        # Je map les features TFT (hidden_size=128) vers les Q-values pour chaque action
        # Linear sans activation car Q-values peuvent etre negatifs (rewards negatifs possibles)
        self.fc_q = nn.Linear(self.tft_predictor.hidden_size, action_size)

    def forward(self, x):
        """
        Je calcule les Q-values pour un etat donne.
        
        Args:
            x: Etat (batch_size, state_size) ou (batch_size, seq_len, state_size)
        
        Returns:
            q_values: (batch_size, action_size) - Q-value pour chaque action
        """
        # Je m'assure que l'input a 3 dimensions (batch, seq, features)
        # DQNAgent envoie souvent (batch, features) donc j'ajoute la dimension seq
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)

        # Je passe par le TFT pour extraire les features temporelles
        # TFT.lstm retourne (output, (hidden, cell))
        # output contient les hidden states pour tous les pas de temps
        out, _ = self.tft.lstm(x)
        
        # Je prends seulement le dernier hidden state
        # ATTENTION: Si LSTM bidirectionnel, out.shape = (batch, seq, hidden_size*2)
        # Si LSTM unidirectionnel, out.shape = (batch, seq, hidden_size)
        # Je prends le dernier pas de temps qui contient toute l'info aggregee
        out = out[:, -1, :]
        
        # IMPORTANT: Si SimpleTFT utilise bidirectionnel=True, out.shape = (batch, hidden_size*2)
        # Donc fc_q doit avoir input_size = hidden_size*2 au lieu de hidden_size
        # Je verifie si le modele TFT est bidirectionnel
        if hasattr(self.tft, 'lstm'):
            is_bidirectional = self.tft.lstm.bidirectional if hasattr(self.tft.lstm, 'bidirectional') else False
            expected_size = self.tft_predictor.hidden_size * (2 if is_bidirectional else 1)
            
            # Je recree fc_q avec la bonne dimension si necessaire
            if self.fc_q.in_features != expected_size:
                self.fc_q = nn.Linear(expected_size, self.fc_q.out_features).to(x.device)
        
        # Je calcule les Q-values pour chaque action
        # Pas d'activation car Q-learning necessite des valeurs non-bornees
        q_values = self.fc_q(out)
        
        return q_values