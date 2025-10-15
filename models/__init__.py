"""
Module de prédiction pour le trading.

Contient les différents modèles de prédiction:
- LSTM: Prédicteur basé sur LSTM (maintenant)
- TFT: Temporal Fusion Transformer (futur)
- LNN: Liquid Neural Network (futur)

Utilisation:
    from models import LSTMPredictor
    
    predictor = LSTMPredictor(sequence_length=60)
    X, y = predictor.prepare_data(df)
    predictor.train(X_train, y_train, X_val, y_val)
    predictions = predictor.predict(X_test)
"""

from .base_predictor import BasePredictor
from .lstm_predictor import LSTMPredictor, LSTMModel

__all__ = [
    'BasePredictor',
    'LSTMPredictor',
    'LSTMModel'
]

__version__ = '1.0.0'