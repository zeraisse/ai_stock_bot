"""
Formateur de donnees pour TFT (Temporal Fusion Transformer).

Je cree ce module pour convertir mes donnees CSV en format TimeSeriesDataSet
que TFT peut comprendre. TFT a besoin de trois types de variables :
- Variables statiques (ex: symbole boursier)
- Variables temporelles observees (ex: prix passe, non connues dans futur)
- Variables temporelles connues (ex: jour de la semaine, connues a l'avance)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings('ignore')


class TFTDataFormatter:
    """
    Je cree cette classe pour formater les donnees boursieres pour TFT.
    
    TFT a besoin d'un format specifique avec :
    - time_idx : index temporel (0, 1, 2, ...)
    - group_ids : identifiant du groupe (symbole boursier)
    - target : variable a predire (prix)
    - Variables temporelles diverses
    """
    
    def __init__(self,
                 max_encoder_length: int = 60,
                 max_prediction_length: int = 1,
                 target_column: str = 'close'):
        """
        J'initialise le formateur avec les parametres temporels.
        
        Args:
            max_encoder_length: Longueur de la fenetre passee (60 jours)
            max_prediction_length: Horizon de prediction (1 jour)
            target_column: Colonne a predire (close)
        """
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.target_column = target_column
        
        print(f"Je cree le formateur TFT avec :")
        print(f"  - Fenetre passee : {max_encoder_length} jours")
        print(f"  - Horizon futur : {max_prediction_length} jour")
        print(f"  - Cible : {target_column}")
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        J'ajoute des features temporelles que TFT peut utiliser.
        
        Ces features sont "connues a l'avance" car on connait toujours
        quel jour/mois on sera demain.
        
        Args:
            df: DataFrame avec colonne Date
            
        Returns:
            df: DataFrame avec features temporelles ajoutees
        """
        df = df.copy()
        
        # Je convertis la date en datetime si ce n'est pas deja fait
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # J'extrais les composantes temporelles
        df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Lundi, 6=Dimanche
        df['day_of_month'] = df['Date'].dt.day
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Je cree un flag pour les jours de trading speciaux
        # (debut/fin de mois, debut/fin d'annee)
        df['is_month_start'] = (df['Date'].dt.day <= 5).astype(int)
        df['is_month_end'] = (df['Date'].dt.day >= 25).astype(int)
        
        print(f"J'ai ajoute 7 features temporelles connues")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        J'ajoute des indicateurs techniques comme features observees.
        
        Ces features sont "observees" car on ne connait pas leur valeur future.
        Je reprends les memes indicateurs que dans LSTM.
        
        Args:
            df: DataFrame avec colonnes OHLCV
            
        Returns:
            df: DataFrame avec indicateurs techniques
        """
        df = df.copy()
        
        # Moyennes mobiles
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Je remplis les valeurs manquantes
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        print(f"J'ai ajoute 11 indicateurs techniques")
        
        return df
    
    def prepare_dataframe(self, 
                         df: pd.DataFrame,
                         symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Je prepare le DataFrame pour TFT.
        
        Je dois creer :
        - time_idx : index temporel incrementiel
        - group_ids : identifiant du symbole
        - Toutes les features necessaires
        
        Args:
            df: DataFrame brut avec colonnes [open, high, low, close, volume]
            symbol: Symbole boursier (optionnel, utilise 'STOCK' par defaut)
            
        Returns:
            df_prepared: DataFrame pret pour TimeSeriesDataSet
        """
        df = df.copy()
        
        # Je verifie que les colonnes essentielles existent
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")
        
        # J'ajoute une colonne Date si elle n'existe pas
        if 'Date' not in df.columns:
            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # J'ajoute les features temporelles
        df = self.add_time_features(df)
        
        # J'ajoute les indicateurs techniques
        df = self.add_technical_indicators(df)
        
        # Je cree l'index temporel (obligatoire pour TFT)
        df['time_idx'] = range(len(df))
        
        # Je cree l'identifiant de groupe (symbole)
        df['group_id'] = symbol if symbol else 'STOCK'
        
        # Je m'assure que le target est present
        if self.target_column not in df.columns:
            raise ValueError(f"Colonne target '{self.target_column}' introuvable")
        
        # Je renomme la colonne target pour uniformite
        df['target'] = df[self.target_column]
        
        print(f"DataFrame prepare avec {len(df)} lignes et {len(df.columns)} colonnes")
        
        return df
    
    def create_timeseries_dataset(self,
                                 df: pd.DataFrame,
                                 training: bool = True) -> TimeSeriesDataSet:
        """
        Je cree le TimeSeriesDataSet pour TFT.
        
        C'est le format que pytorch-forecasting comprend. Je specifie :
        - Quelles variables sont statiques
        - Quelles variables sont observees (inconnues dans futur)
        - Quelles variables sont connues (connues dans futur)
        
        Args:
            df: DataFrame prepare
            training: Si True, c'est pour l'entrainement (sinon validation/test)
            
        Returns:
            dataset: TimeSeriesDataSet pret pour TFT
        """
        # Je definis les colonnes par type
        # Variables STATIQUES : ne changent pas dans le temps (symbole)
        static_categoricals = ['group_id']
        
        # Variables TEMPORELLES CONNUES : on connait leur valeur future
        # (jour de la semaine, mois, etc.)
        time_varying_known_categoricals = []
        time_varying_known_reals = [
            'time_idx',  # Obligatoire
            'day_of_week',
            'day_of_month', 
            'week_of_year',
            'month',
            'quarter',
            'is_month_start',
            'is_month_end'
        ]
        
        # Variables TEMPORELLES OBSERVEES : on ne connait PAS leur valeur future
        # (prix, volume, indicateurs techniques)
        time_varying_unknown_reals = [
            'open', 'high', 'low', 'close', 'volume',
            'ma_7', 'ma_21', 'rsi', 'macd', 'macd_signal',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_ma', 'volume_ratio',
            'target'  # La cible est aussi une variable observee
        ]
        
        # Je cree le dataset
        dataset = TimeSeriesDataSet(
            df,
            time_idx='time_idx',
            target='target',
            group_ids=static_categoricals,
            
            # Je definis les longueurs temporelles
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            
            # Je specifie les types de variables
            static_categoricals=static_categoricals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            
            # Je normalise les variables continues
            target_normalizer=GroupNormalizer(
                groups=static_categoricals,
                transformation='softplus'  # Transformation pour valeurs positives
            ),
            
            # J'autorise les sequences incompletes au debut
            min_encoder_length=self.max_encoder_length // 2,
            
            # Je ne predis que si j'ai assez de donnees passees
            allow_missing_timesteps=False,
        )
        
        mode = "entrainement" if training else "validation/test"
        print(f"TimeSeriesDataSet cree pour {mode} avec {len(dataset)} sequences")
        
        return dataset
    
    def get_feature_list(self) -> dict:
        """
        Je retourne la liste de toutes les features par categorie.
        
        Returns:
            dict: Dictionnaire avec les listes de features
        """
        return {
            'static': ['group_id'],
            'known': [
                'time_idx', 'day_of_week', 'day_of_month', 
                'week_of_year', 'month', 'quarter',
                'is_month_start', 'is_month_end'
            ],
            'observed': [
                'open', 'high', 'low', 'close', 'volume',
                'ma_7', 'ma_21', 'rsi', 'macd', 'macd_signal',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
                'volume_ma', 'volume_ratio', 'target'
            ],
            'target': 'target'
        }


# Fonction utilitaire pour tester le formateur
def test_formatter():
    """
    Je teste le formateur sur des donnees factices.
    """
    print("Test du formateur TFT...")
    print("="*60)
    
    # Je cree des donnees factices
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    df_test = pd.DataFrame({
        'Date': dates,
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 200)
    })
    
    # Je cree le formateur
    formatter = TFTDataFormatter(max_encoder_length=60)
    
    # Je prepare les donnees
    df_prepared = formatter.prepare_dataframe(df_test, symbol='TEST')
    
    # Je cree le dataset
    dataset = formatter.create_timeseries_dataset(df_prepared, training=True)
    
    print("\nTest reussi!")
    print(f"Features disponibles : {formatter.get_feature_list()}")
    
    return formatter, dataset


if __name__ == "__main__":
    test_formatter()