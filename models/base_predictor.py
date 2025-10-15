"""
Interface abstraite pour tous les prédicteurs (LSTM, TFT, LNN).
Permet de changer facilement de modèle sans modifier le reste du code.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import torch


class BasePredictor(ABC):
    """
    Interface abstraite pour tous les modèles de prédiction.
    
    Architecture modulaire permettant de passer facilement de:
    - LSTM → TFT → TFT+LNN
    
    Tous les prédicteurs doivent implémenter ces méthodes.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_features: int = 3,
                 device: str = 'cuda'):
        """
        Initialiser le prédicteur.
        
        Args:
            sequence_length: Nombre de pas de temps passés (fenêtre)
            prediction_features: Nombre de features à prédire
            device: 'cuda' ou 'cpu'
        """
        self.sequence_length = sequence_length
        self.prediction_features = prediction_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Sera initialisé lors de prepare_data
        self.scaler = None
        self.feature_columns = None
        
    @abstractmethod
    def build_model(self, input_size: int) -> torch.nn.Module:
        """
        Construire l'architecture du modèle.
        
        Args:
            input_size: Nombre de features en entrée
            
        Returns:
            model: Modèle PyTorch
        """
        pass
    
    @abstractmethod
    def prepare_data(self, 
                     df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Préparer les données pour l'entraînement.
        
        Args:
            df: DataFrame avec colonnes [open, high, low, close, volume]
            
        Returns:
            X: Séquences d'entrée (n_samples, sequence_length, n_features)
            y: Cibles (n_samples, prediction_features)
        """
        pass
    
    @abstractmethod
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001) -> Dict:
        """
        Entraîner le modèle.
        
        Args:
            X_train: Données d'entraînement
            y_train: Cibles d'entraînement
            X_val: Données de validation
            y_val: Cibles de validation
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            learning_rate: Taux d'apprentissage
            
        Returns:
            history: Historique d'entraînement (loss, metrics)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faire des prédictions.
        
        Args:
            X: Séquences d'entrée (n_samples, sequence_length, n_features)
            
        Returns:
            predictions: Prédictions (n_samples, prediction_features)
        """
        pass
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédire directement depuis un DataFrame.
        
        Args:
            df: DataFrame avec données
            
        Returns:
            predictions: Array de prédictions
        """
        X, _ = self.prepare_data(df)
        return self.predict(X)
    
    @abstractmethod
    def save(self, path: str):
        """
        Sauvegarder le modèle.
        
        Args:
            path: Chemin de sauvegarde (ex: 'saved_models/lstm_AAPL.pth')
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Charger le modèle.
        
        Args:
            path: Chemin du modèle sauvegardé
        """
        pass
    
    def get_info(self) -> Dict:
        """
        Obtenir les informations du modèle.
        
        Returns:
            info: Dictionnaire avec infos du modèle
        """
        return {
            'type': self.__class__.__name__,
            'sequence_length': self.sequence_length,
            'prediction_features': self.prediction_features,
            'device': str(self.device),
            'num_parameters': self._count_parameters() if hasattr(self, 'model') else 0
        }
    
    def _count_parameters(self) -> int:
        """Compter le nombre de paramètres du modèle."""
        if not hasattr(self, 'model'):
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Évaluer le modèle sur des données de test.
        
        Args:
            X_test: Données de test
            y_test: Cibles de test
            
        Returns:
            metrics: Dictionnaire avec métriques (MSE, MAE, etc.)
        """
        predictions = self.predict(X_test)
        
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # Calculer par feature
        feature_names = ['price', 'trend', 'volatility']
        feature_metrics = {}
        
        for i, name in enumerate(feature_names[:y_test.shape[1]]):
            feature_metrics[f'{name}_mae'] = np.mean(np.abs(predictions[:, i] - y_test[:, i]))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            **feature_metrics
        }