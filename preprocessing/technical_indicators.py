"""
Calcul des indicateurs techniques pour l'analyse financière.

Indicateurs implémentés:
- Moving Averages (MA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Calculateur d'indicateurs techniques pour les données financières.
    
    Utilisation:
        ti = TechnicalIndicators()
        df_with_indicators = ti.add_all_indicators(df)
    """
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame,
                           windows: list = [7, 21, 50],
                           column: str = 'close') -> pd.DataFrame:
        """
        Ajouter des moyennes mobiles.
        
        Args:
            df: DataFrame avec données
            windows: Liste des fenêtres temporelles
            column: Colonne à utiliser
            
        Returns:
            df: DataFrame avec MA ajoutées
        """
        df = df.copy()
        
        for window in windows:
            df[f'ma_{window}'] = df[column].rolling(window=window).mean()
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame,
                window: int = 14,
                column: str = 'close') -> pd.DataFrame:
        """
        Ajouter le Relative Strength Index (RSI).
        
        Args:
            df: DataFrame avec données
            window: Période de calcul
            column: Colonne à utiliser
            
        Returns:
            df: DataFrame avec RSI ajouté
        """
        df = df.copy()
        
        # Calculer les variations
        delta = df[column].diff()
        
        # Séparer gains et pertes
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Calculer RS et RSI
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9,
                 column: str = 'close') -> pd.DataFrame:
        """
        Ajouter le MACD et son signal.
        
        Args:
            df: DataFrame avec données
            fast: Période EMA rapide
            slow: Période EMA lente
            signal: Période du signal
            column: Colonne à utiliser
            
        Returns:
            df: DataFrame avec MACD ajouté
        """
        df = df.copy()
        
        # Calculer les EMA
        exp1 = df[column].ewm(span=fast, adjust=False).mean()
        exp2 = df[column].ewm(span=slow, adjust=False).mean()
        
        # MACD et signal
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame,
                           window: int = 20,
                           num_std: float = 2.0,
                           column: str = 'close') -> pd.DataFrame:
        """
        Ajouter les Bandes de Bollinger.
        
        Args:
            df: DataFrame avec données
            window: Période de calcul
            num_std: Nombre d'écarts-types
            column: Colonne à utiliser
            
        Returns:
            df: DataFrame avec Bollinger Bands ajoutées
        """
        df = df.copy()
        
        # Bande centrale (moyenne mobile)
        df['bb_middle'] = df[column].rolling(window=window).mean()
        
        # Écart-type
        bb_std = df[column].rolling(window=window).std()
        
        # Bandes supérieure et inférieure
        df['bb_upper'] = df['bb_middle'] + (bb_std * num_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * num_std)
        
        # Position dans les bandes (0 = bande basse, 1 = bande haute)
        df['bb_position'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Largeur des bandes (volatilité)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajouter des indicateurs de volume.
        
        Args:
            df: DataFrame avec colonne 'volume'
            
        Returns:
            df: DataFrame avec indicateurs de volume ajoutés
        """
        df = df.copy()
        
        # Changement de volume en %
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moyen mobile
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # Ratio volume / moyenne
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame,
                               column: str = 'close') -> pd.DataFrame:
        """
        Ajouter des indicateurs de momentum.
        
        Args:
            df: DataFrame avec données
            column: Colonne à utiliser
            
        Returns:
            df: DataFrame avec momentum ajouté
        """
        df = df.copy()
        
        # Rate of Change (ROC)
        df['roc'] = df[column].pct_change(periods=10) * 100
        
        # Momentum
        df['momentum'] = df[column] - df[column].shift(10)
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame,
                                  window: int = 14,
                                  column: str = 'close') -> pd.DataFrame:
        """
        Ajouter des indicateurs de volatilité.
        
        Args:
            df: DataFrame avec données
            window: Période de calcul
            column: Colonne à utiliser
            
        Returns:
            df: DataFrame avec volatilité ajoutée
        """
        df = df.copy()
        
        # Écart-type (volatilité)
        df['volatility'] = df[column].rolling(window=window).std()
        
        # Volatilité normalisée
        df['volatility_normalized'] = df['volatility'] / df[column].rolling(window=window).mean()
        
        # Average True Range (ATR)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=window).mean()
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame,
                          ma_windows: list = [7, 21],
                          rsi_window: int = 14,
                          bb_window: int = 20) -> pd.DataFrame:
        """
        Ajouter tous les indicateurs techniques.
        
        Args:
            df: DataFrame avec colonnes [open, high, low, close, volume]
            ma_windows: Fenêtres pour les moyennes mobiles
            rsi_window: Période pour le RSI
            bb_window: Période pour les Bollinger Bands
            
        Returns:
            df: DataFrame avec tous les indicateurs
        """
        df = df.copy()
        
        # Add all indicateurs
        df = TechnicalIndicators.add_moving_averages(df, windows=ma_windows)
        df = TechnicalIndicators.add_rsi(df, window=rsi_window)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df, window=bb_window)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_momentum_indicators(df)
        df = TechnicalIndicators.add_volatility_indicators(df)
        
        # Remplir les NaN
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    @staticmethod
    def get_feature_list() -> list:
        """
        Obtenir la liste de tous les indicateurs techniques.
        
        Returns:
            features: Liste des noms de features
        """
        return [
            # Prix de base
            'open', 'high', 'low', 'close', 'volume',
            # Moving Averages
            'ma_7', 'ma_21',
            # RSI
            'rsi',
            # MACD
            'macd', 'macd_signal', 'macd_histogram',
            # Bollinger Bands
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
            # Volume
            'volume_change', 'volume_ma_20', 'volume_ratio',
            # Momentum
            'roc', 'momentum',
            # Volatility
            'volatility', 'volatility_normalized', 'atr'
        ]