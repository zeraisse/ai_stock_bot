"""
Prédicteur LSTM pour les prix d'actions.

Prédit 3 features:
1. Prix futur (close price +1 jour)
2. Tendance (-1=baisse, 0=stable, +1=hausse)
3. Volatilité (écart-type sur 5 jours)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional
import pickle

from .base_predictor import BasePredictor


# MODÈLE LSTM


class LSTMModel(nn.Module):
    """
    Architecture LSTM avec attention pour la prédiction de prix.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Nombre de features en entrée
            hidden_size: Taille des couches cachées
            num_layers: Nombre de couches LSTM
            output_size: Nombre de prédictions (3: prix, tendance, volatilité)
            dropout: Taux de dropout
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass avec attention.
        
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            predictions: (batch_size, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Contexte pondéré
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)
        
        # Prédictions
        predictions = self.fc(context)
        
        return predictions


# PRÉDICTEUR LSTM

class LSTMPredictor(BasePredictor):
    """
    Prédicteur basé sur LSTM pour les marchés financiers.
    
    Utilisation:
        predictor = LSTMPredictor(sequence_length=60)
        X, y = predictor.prepare_data(df)
        predictor.train(X_train, y_train, X_val, y_val)
        predictions = predictor.predict(X_test)
    """
    
    def __init__(self,
                 sequence_length: int = 60,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 device: str = 'cuda'):
        """
        Args:
            sequence_length: Longueur de la fenêtre temporelle
            hidden_size: Taille des couches cachées LSTM
            num_layers: Nombre de couches LSTM
            dropout: Taux de dropout
            device: 'cuda' ou 'cpu'
        """
        super().__init__(
            sequence_length=sequence_length,
            prediction_features=3,
            device=device
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_size: int) -> nn.Module:
        """Construire le modèle LSTM."""
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.prediction_features,
            dropout=self.dropout
        )
        return model.to(self.device)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajouter des indicateurs techniques au DataFrame.
        
        Args:
            df: DataFrame avec colonnes [open, high, low, close, volume]
            
        Returns:
            df_with_indicators: DataFrame avec indicateurs ajoutés
        """
        df = df.copy()
        
        # 1. Moving Averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        
        # 2. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 4. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 5. Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Remplir les NaN
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Préparer les données pour l'entraînement LSTM.
        
        Args:
            df: DataFrame avec colonnes [open, high, low, close, volume]
            
        Returns:
            X: Séquences (n_samples, sequence_length, n_features)
            y: Cibles (n_samples, 3) [prix_futur, tendance, volatilité]
        """
        # adding indicateurs techniques
        df_indicators = self._add_technical_indicators(df)
        
        # Colonnes à utiliser comme features
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ma_7', 'ma_21', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'volume_change'
        ]
        
        # Normaliser les features
        features_data = df_indicators[self.feature_columns].values
        features_normalized = self.scaler.fit_transform(features_data)
        
        # Créer les séquences
        X = []
        y = []
        
        for i in range(self.sequence_length, len(df_indicators) - 1):
            # Séquence d'entrée (60 derniers jours)
            X.append(features_normalized[i - self.sequence_length:i])
            
            # Cibles (jour suivant)
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + 1]
            
            # 1. Prix futur (normalisé)
            # On utilise le ratio pour préserver l'information
            price_ratio = future_price / current_price
            
            # 2. Tendance (-1, 0, 1)
            price_change_pct = (future_price - current_price) / current_price
            if price_change_pct > 0.001:  # Hausse > 0.1%
                trend = 1.0
            elif price_change_pct < -0.001:  # Baisse > 0.1%
                trend = -1.0
            else:
                trend = 0.0
            
            # 3. Volatilité (écart-type des 5 derniers jours, normalisé)
            recent_prices = df['close'].iloc[i-5:i].values
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            y.append([price_ratio, trend, volatility])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001) -> Dict:
        """
        Entraîner le modèle LSTM.
        
        Args:
            X_train, y_train: Données d'entraînement
            X_val, y_val: Données de validation
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            learning_rate: Taux d'apprentissage
            
        Returns:
            history: Historique d'entraînement
        """
        # Construire le modèle
        input_size = X_train.shape[2]
        self.model = self.build_model(input_size)
        
        # Créer les DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer et loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        
        # Historique
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT DU MODÈLE LSTM")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # === TRAINING ===
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # === VALIDATION ===
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Enregistrer
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch'].append(epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping à l'époque {epoch}")
                break
            
            # Afficher progression
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Best: {best_val_loss:.6f}")
        
        print(f"\n{'='*60}")
        print(f"✓ ENTRAÎNEMENT TERMINÉ")
        print(f"Meilleure val loss: {best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faire des prédictions.
        
        Args:
            X: Séquences (n_samples, sequence_length, n_features)
            
        Returns:
            predictions: (n_samples, 3) [prix_ratio, tendance, volatilité]
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné. Appelez train() d'abord.")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Traiter par batch pour efficacité
            batch_size = 128
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                pred = self.model(batch_tensor)
                predictions.append(pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    def save(self, path: str):
        """
        Sauvegarder le modèle et le scaler.
        
        Args:
            path: Chemin (ex: 'saved_models/lstm_AAPL.pth')
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'prediction_features': self.prediction_features
            }
        }, path, _use_new_zipfile_serialization=True)
        
        print(f"Nice Hakim Modèle sauvegardé: {path}")
    
    def load(self, path: str):
        """
        Charger un modèle sauvegardé.
        
        Args:
            path: Chemin du modèle
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Charger la config
        config = checkpoint['config']
        self.sequence_length = config['sequence_length']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.prediction_features = config['prediction_features']
        
        # Charger le scaler
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        
        # Reconstruire le modèle
        input_size = len(self.feature_columns)
        self.model = self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Modèle chargé: {path}")