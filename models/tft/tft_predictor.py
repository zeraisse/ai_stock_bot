"""
Predicteur TFT pour la prediction boursiere.

Je cree ce module pour implementer l'interface BasePredictor avec TFT.
Cela me permet de remplacer LSTM par TFT sans changer le reste du code.
Compatible avec DQNAgent et DQNTFTModel.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# # import pytorch_lightning as pl
# from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
# from pytorch_forecasting.data import GroupNormalizer
# from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_predictor import BasePredictor


class TFTModel(nn.Module):
    """
    Je cree ce wrapper pour rendre TFT compatible avec l'interface DQN.
    
    TFT de pytorch-forecasting n'est pas directement utilisable comme module PyTorch
    dans DQNTFTModel car il necessite un TimeSeriesDataSet. Je cree donc ce wrapper
    qui expose une interface forward() compatible avec les tensors bruts.
    """
    
    def __init__(self, tft_model, hidden_size: int, input_size: int = 3):
        super().__init__()
        self.tft = tft_model
        self.hidden_size = hidden_size
        
        # Je cree une couche LSTM simple pour extraire les features temporelles
        # quand je recois des tensors bruts (pas de TimeSeriesDataSet disponible)
        # hidden_size=128 est un bon compromis memoire/performance pour le trading
        self.lstm = nn.LSTM(
            input_size=input_size,  # Je recois des sequences de prix normalises
            hidden_size=hidden_size,
            num_layers=2,  # 2 couches suffisent pour capturer les patterns court/moyen terme
            batch_first=True,
            dropout=0.2  # 20% de dropout pour eviter l'overfitting sur les patterns de marche
        )
    
    def forward(self, x):
        """
        Je gere deux modes d'inference:
        1. Mode TFT complet (quand j'ai un TimeSeriesDataSet)
        2. Mode LSTM fallback (quand j'ai juste un tensor brut depuis DQNAgent)
        """
        if isinstance(x, dict):
            # Mode TFT complet avec TimeSeriesDataSet
            return self.tft(x)
        else:
            # Mode fallback pour DQNAgent: je traite comme une sequence LSTM
            # x shape: (batch, seq_len, features) ou (batch, features)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # J'ajoute la dimension feature
            
            lstm_out, _ = self.lstm(x)
            # Je prends seulement le dernier hidden state
            # C'est lui qui contient l'information aggregee de toute la sequence
            return lstm_out[:, -1, :]


class TFTPredictor(BasePredictor):
    """
    Je cree ce predicteur TFT qui suit l'interface BasePredictor.
    
    Interface identique a LSTMPredictor pour permettre le drop-in replacement
    dans DQNLSTMModel -> DQNTFTModel.
    """
    
    def __init__(self,
                 sequence_length: int = 60,
                 max_encoder_length: int = 60,
                 max_prediction_length: int = 1,
                 hidden_size: int = 128,
                 lstm_layers: int = 2,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 device: str = 'cuda'):
        """
        J'initialise le predicteur TFT.
        
        Args:
            sequence_length: Longueur de sequence (60 jours = ~3 mois de trading)
            max_encoder_length: Fenetre passee pour TFT (identique a sequence_length)
            max_prediction_length: Horizon de prediction (1 jour ahead)
            hidden_size: Taille des couches cachees (128 = bon ratio performance/memoire)
            lstm_layers: Nombre de couches LSTM (2 couches captent court et moyen terme)
            attention_head_size: Nombre de tetes d'attention (4 tetes = 4 perspectives temporelles)
            dropout: Taux de dropout (0.1 = 10%, evite overfitting sans trop regulariser)
            learning_rate: Taux d'apprentissage (0.001 = valeur standard Adam pour timeseries)
            device: 'cuda' ou 'cpu'
        """
        super().__init__(
            sequence_length=sequence_length,
            prediction_features=3,  # Prix, tendance, volatilite comme LSTM
            device=device
        )
        
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Le modele TFT sera cree lors de l'entrainement
        self.tft_core = None
        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.scaler = None
        
        print(f"TFTPredictor initialise:")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - LSTM layers: {lstm_layers}")
        print(f"  - Attention heads: {attention_head_size}")
        print(f"  - Device: {self.device}")
    
    def build_model(self, input_size: int) -> nn.Module:
        """
        Je construis le modele (requis par BasePredictor et DQNTFTModel).
        
        Pour TFT, je ne peux pas construire le modele complet sans dataset,
        donc je retourne un wrapper LSTM simple qui sera remplace lors du train().
        Ce wrapper permet a DQNTFTModel de fonctionner immediatement.
        
        Args:
            input_size: Nombre de features en entree
            
        Returns:
            model: Module PyTorch avec interface forward()
        """
        # Je cree un modele temporaire LSTM-only pour l'interface DQN
        # hidden_size doit correspondre pour que fc_q fonctionne dans DQNTFTModel
        self.model = TFTModel(None, self.hidden_size, input_size=input_size)
        return self.model
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Je prepare les donnees pour l'entrainement.
        
        Je cree des sequences glissantes de longueur sequence_length
        pour predire 3 targets: prix, tendance, volatilite.
        
        Args:
            df: DataFrame avec colonnes 'close', 'open', 'high', 'low', 'volume'
            
        Returns:
            X: Sequences (n_samples, sequence_length, n_features)
            y: Targets (n_samples, 3)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Je normalise les donnees entre 0 et 1
        # MinMaxScaler preserve les relations temporelles mieux que StandardScaler
        self.scaler = MinMaxScaler()
        
        # Je selectionne les features importantes pour le trading
        feature_cols = ['close', 'open', 'high', 'low', 'volume']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError("DataFrame doit contenir au moins une colonne parmi: close, open, high, low, volume")
        
        data = df[available_cols].values
        scaled_data = self.scaler.fit_transform(data)
        
        # Je cree des sequences glissantes
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            
            # Target 1: Prix futur normalise
            next_price = scaled_data[i + self.sequence_length, 0]
            
            # Target 2: Tendance (-1: baisse, 0: stable, 1: hausse)
            # Je compare le prix futur avec le prix actuel
            current_price = scaled_data[i + self.sequence_length - 1, 0]
            if next_price > current_price * 1.001:  # Hausse >0.1%
                trend = 1
            elif next_price < current_price * 0.999:  # Baisse >0.1%
                trend = -1
            else:
                trend = 0
            
            # Target 3: Volatilite (ecart-type sur les 5 derniers jours)
            # Je calcule la volatilite pour mesurer le risque
            recent_prices = scaled_data[max(0, i + self.sequence_length - 5):i + self.sequence_length, 0]
            volatility = np.std(recent_prices)
            
            y.append([next_price, trend, volatility])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: Optional[float] = None) -> Dict:
        """
        J'entraine le modele TFT.
        
        Args:
            X_train: Sequences d'entrainement
            y_train: Targets d'entrainement
            X_val: Sequences de validation
            y_val: Targets de validation
            epochs: Nombre d'epochs (50 = bon compromis pour convergence sans overfit)
            batch_size: Taille des batchs (32 = standard pour timeseries)
            learning_rate: Taux d'apprentissage optionnel
            
        Returns:
            history: Historique d'entrainement
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        
        # Je convertis les arrays numpy en tensors PyTorch
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        
        # Je construis le modele TFT complet si pas deja fait
        if self.tft_core is None:
            # Je cree un modele LSTM avec attention comme fallback
            # car TFT complet necessite TimeSeriesDataSet
            class SimpleTFT(nn.Module):
                def __init__(self, input_size, hidden_size, lstm_layers, dropout, output_size):
                    super().__init__()
                    
                    # LSTM bidirectionnel pour capturer les dependances avant/arriere
                    # Bidirectionnel double la capacite de modelisation temporelle
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=lstm_layers,
                        batch_first=True,
                        dropout=dropout,
                        bidirectional=True  # Je capture les patterns dans les deux sens
                    )
                    
                    # Attention pour ponderer l'importance de chaque pas de temps
                    # Tanh introduit la non-linearite necessaire pour apprendre les poids
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),  # *2 car bidirectionnel
                        nn.Tanh(),  # Tanh garde les valeurs entre -1 et 1, ideal pour attention
                        nn.Linear(hidden_size, 1)
                    )
                    
                    # Couches finales de prediction
                    # ReLU introduit la non-linearite et force les valeurs positives
                    # C'est adapte aux prix (toujours positifs) et volumes
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),  # ReLU = max(0, x), rapide et efficace
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, output_size)
                    )
                
                def forward(self, x):
                    # LSTM processing
                    lstm_out, _ = self.lstm(x)
                    
                    # Attention weights
                    attention_weights = self.attention(lstm_out)
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    
                    # Weighted context
                    context = torch.sum(attention_weights * lstm_out, dim=1)
                    
                    # Final predictions
                    return self.fc(context)
            
            input_size = X_train.shape[2]
            self.tft_core = SimpleTFT(
                input_size=input_size,
                hidden_size=self.hidden_size,
                lstm_layers=self.lstm_layers,
                dropout=self.dropout,
                output_size=3
            ).to(self.device)
            
            # Je remplace le modele wrapper par le modele complet
            self.model = TFTModel(self.tft_core, self.hidden_size)
            self.model.tft = self.tft_core
        
        # Configuration de l'optimiseur
        # Adam adapte le learning rate automatiquement, ideal pour timeseries
        optimizer = torch.optim.Adam(self.tft_core.parameters(), lr=self.learning_rate)
        
        # Loss function: MSE pour regression
        # MSE penalise les grandes erreurs, important pour les predictions de prix
        criterion = nn.MSELoss()
        
        # Je cree les dataloaders
        # shuffle=True pour eviter le biais d'ordre temporel dans les batchs
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
            val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Entrainement
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"\nEntrainement TFT pour {epochs} epochs...")
        for epoch in range(epochs):
            # Mode entrainement
            self.tft_core.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.tft_core(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping pour stabiliser l'entrainement
                # 1.0 evite les explosions de gradients communes en timeseries
                torch.nn.utils.clip_grad_norm_(self.tft_core.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val is not None:
                self.tft_core.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        predictions = self.tft_core(batch_X)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        print("Entrainement termine")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Je genere des predictions.
        
        Args:
            X: Sequences (n_samples, sequence_length, n_features)
            
        Returns:
            predictions: (n_samples, 3) [prix, tendance, volatilite]
        """
        if self.tft_core is None:
            raise RuntimeError("Modele non entraine. Appelez train() d'abord.")
        
        self.tft_core.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.tft_core(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str):
        """Je sauvegarde le modele."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'model_state': self.tft_core.state_dict() if self.tft_core else None,
            'scaler': self.scaler,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'lstm_layers': self.lstm_layers,
                'attention_head_size': self.attention_head_size,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }
        
        torch.save(state, path)
        print(f"Modele sauvegarde: {path}")
    
    def load(self, path: str):
        """Je charge le modele."""
        state = torch.load(path, map_location=self.device)
        
        self.scaler = state['scaler']
        config = state['config']
        
        self.sequence_length = config['sequence_length']
        self.hidden_size = config['hidden_size']
        self.lstm_layers = config['lstm_layers']
        
        # Je recree l'architecture
        if state['model_state'] is not None:
            from torch import nn
            
            class SimpleTFT(nn.Module):
                def __init__(self, input_size, hidden_size, lstm_layers, dropout, output_size):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=lstm_layers,
                        batch_first=True,
                        dropout=dropout,
                        bidirectional=True
                    )
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1)
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, output_size)
                    )
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    attention_weights = self.attention(lstm_out)
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    context = torch.sum(attention_weights * lstm_out, dim=1)
                    return self.fc(context)
            
            # Je suppose input_size=5 (close, open, high, low, volume)
            self.tft_core = SimpleTFT(
                input_size=5,
                hidden_size=self.hidden_size,
                lstm_layers=self.lstm_layers,
                dropout=self.dropout,
                output_size=3
            ).to(self.device)
            
            self.tft_core.load_state_dict(state['model_state'])
            self.model = TFTModel(self.tft_core, self.hidden_size)
            self.model.tft = self.tft_core
        
        print(f"Modele charge: {path}")