"""
Module de prétraitement des données pour le trading.

Contient:
- DataProcessor: Préparation et normalisation des données
- TechnicalIndicators: Calcul des indicateurs techniques

Utilisation:
    from preprocessing import DataProcessor, TechnicalIndicators
    
    processor = DataProcessor(sequence_length=60)
    X, y = processor.process(df)
"""

from .data_processor import DataProcessor
from .technical_indicators import TechnicalIndicators

__all__ = [
    'DataProcessor',
    'TechnicalIndicators'
]

__version__ = '1.0.0'