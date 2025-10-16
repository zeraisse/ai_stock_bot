"""
Processeur de données pour le trading.

Responsabilités:
- Normalisation des données
- Création des séquences temporelles
- Split train/val/test
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Processeur de données pour les modèles de prédiction.
    
    Utilisation:
        processor = DataProcessor(sequence_length=60)
        X, y = processor.create_sequences(df)
        X_train, X_val, X_test = processor.split_data(X, train=0.7, val=0.15)
    """
    
    def __init__(self,
                 sequence_length: int = 60,
                 scaler_type: str = 'minmax',
                 target_column: str = 'close'):
        """
        Args:
            sequence_length: Longueur de la fenêtre temporelle
            scaler_type: 'minmax' ou 'standard'
            target_column: Colonne à prédire
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        # Choisir le scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"scaler_type doit être 'minmax' ou 'standard', pas '{scaler_type}'")
        
        self.feature_columns = None
        
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normaliser les données.
        
        Args:
            df: DataFrame à normaliser
            fit: Si True, fit le scaler. Si False, utilise le scaler existant
            
        Returns:
            df_normalized: DataFrame normalisé
        """
        if fit:
            normalized_values = self.scaler.fit_transform(df)
        else:
            normalized_values = self.scaler.transform(df)
        
        df_normalized = pd.DataFrame(
            normalized_values,
            columns=df.columns,
            index=df.index
        )
        
        return df_normalized
    
    def create_sequences(self,
                        df: pd.DataFrame,
                        target_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Créer des séquences temporelles pour l'entraînement.
        
        Args:
            df: DataFrame avec features
            target_df: DataFrame avec cibles (optionnel, sinon utilise df)
            
        Returns:
            X: Séquences (n_samples, sequence_length, n_features)
            y: Cibles (n_samples, n_target_features)
        """
        if target_df is None:
            target_df = df
        
        X = []
        y = []
        
        for i in range(self.sequence_length, len(df)):
            # Séquence d'entrée (sequence_length derniers pas de temps)
            X.append(df.iloc[i - self.sequence_length:i].values)
            
            # Cible (pas de temps suivant)
            y.append(target_df.iloc[i].values)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple:
        """
        Diviser les données en train/val/test.
        
        Args:
            X: Séquences d'entrée
            y: Cibles
            train_ratio: Proportion de données d'entraînement
            val_ratio: Proportion de données de validation
            
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n_samples = len(X)#  function train split test integrate 
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Split
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"\n{'='*60}")
        print(f"SPLIT DES DONNÉES")
        print(f"{'='*60}")
        print(f"Total samples:     {n_samples:,}")
        print(f"Train samples:     {len(X_train):,} ({train_ratio*100:.0f}%)")
        print(f"Val samples:       {len(X_val):,} ({val_ratio*100:.0f}%)")
        print(f"Test samples:      {len(X_test):,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
        print(f"{'='*60}\n")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Obtenir l'importance des features (si applicable).
        
        Args:
            feature_names: Liste des noms de features
            
        Returns:
            df_importance: DataFrame avec importance des features
        """
        # Pour l'instant, retourne juste les noms
        # Peut être étendu avec des méthodes d'analyse de features
        return pd.DataFrame({
            'feature': feature_names,
            'index': range(len(feature_names))
        })
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Dénormaliser les données.
        
        Args:
            data: Données normalisées
            
        Returns:
            data_original: Données dénormalisées
        """
        return self.scaler.inverse_transform(data)
    
    def get_info(self) -> Dict:
        """Obtenir les informations du processeur."""
        return {
            'sequence_length': self.sequence_length,
            'scaler_type': type(self.scaler).__name__,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns
        }