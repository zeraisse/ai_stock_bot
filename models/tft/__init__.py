"""
Module TFT (Temporal Fusion Transformer) pour la prediction boursiere.

Je cree ce module pour remplacer LSTM par une architecture plus puissante.
TFT combine attention multi-tete, LSTM et selection automatique de features.

Utilisation:
    from models.tft import TFTPredictor
    
    predictor = TFTPredictor(max_encoder_length=60)
    X, y = predictor.prepare_data(df)
    predictor.train(X_train, y_train, X_val, y_val)
    predictions = predictor.predict(X_test)
"""

from .tft_predictor import TFTPredictor

__all__ = ['TFTPredictor']