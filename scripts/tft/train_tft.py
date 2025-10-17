"""
Script pour entrainer le modele TFT sur des donnees boursieres.

Je cree ce script pour :
1. Charger les donnees boursieres
2. Preparer les donnees au format TFT
3. Entrainer le modele TFT
4. Sauvegarder le modele entraine
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# J'ajoute les chemins vers mes modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from models.tft import TFTPredictor
from env_trading.trading_env import load_stock_data_from_csv, get_available_symbols
CSV_PATH = r'C:\Users\willi\Documents\Ecole_ipssi_MIA4\Octobre_2025\TP_groupe_bourse\ai_stock_bot\dataset\top10_stocks_2025.csv'


def train_tft_model(symbol='AAPL', 
                    epochs=50, 
                    batch_size=64,
                    max_encoder_length=60,
                    save_dir='saved_models'):
    """
    J'entraine un modele TFT sur un symbole boursier.
    
    Args:
        symbol: Symbole boursier (AAPL, TSLA, etc.)
        epochs: Nombre d'epoques d'entrainement
        batch_size: Taille des batchs
        max_encoder_length: Longueur de la fenetre passee
        save_dir: Dossier de sauvegarde
    """
    print("="*80)
    print(f"ENTRAINEMENT TFT POUR {symbol}")
    print("="*80)
    
    # ETAPE 1 : Charger les donnees
    print(f"\n1. Chargement des donnees pour {symbol}...")
    print("-"*80)
    
    csv_path = CSV_PATH

    if not os.path.exists(csv_path):
        print(f"ERREUR : Fichier non trouve : {csv_path}")
        return
    
    try:
        # Je charge les donnees avec ma fonction
        data = load_stock_data_from_csv(csv_path, symbol=symbol)
        print(f"Donnees chargees : {len(data)} jours")
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return
    
    # ETAPE 2 : Creer le predicteur TFT
    print(f"\n2. Creation du predicteur TFT...")
    print("-"*80)
    
    predictor = TFTPredictor(
        max_encoder_length=max_encoder_length,
        max_prediction_length=1,  # Prediction 1 jour dans le futur
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.1,
        learning_rate=0.001
    )
    
    print(f"Predicteur TFT cree avec :")
    print(f"  - Max encoder length : {max_encoder_length}")
    print(f"  - Hidden size : 64")
    print(f"  - LSTM layers : 2")
    print(f"  - Attention heads : 4")
    
    # ETAPE 3 : Preparer les donnees
    print(f"\n3. Preparation des donnees...")
    print("-"*80)
    
    # Je prepare les donnees (cree le TimeSeriesDataSet)
    X, y = predictor.prepare_data(data)
    print(f"Donnees preparees")
    
    # Note : Pour TFT, X et y sont des placeholders
    # Le vrai dataset est stocke dans predictor.training_dataset
    
    # ETAPE 4 : Entrainer le modele
    print(f"\n4. Entrainement du modele TFT...")
    print("-"*80)
    
    try:
        history = predictor.train(
            X_train=X,  # Placeholder (TFT utilise son dataset interne)
            y_train=y,  # Placeholder
            X_val=X,    # Placeholder
            y_val=y,    # Placeholder
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001
        )
        
        print("\nEntrainement termine avec succes !")
        
    except Exception as e:
        print(f"\nERREUR pendant l'entrainement : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ETAPE 5 : Sauvegarder le modele
    print(f"\n5. Sauvegarde du modele...")
    print("-"*80)
    
    # Je cree le dossier de sauvegarde si besoin
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f'tft_{symbol.lower()}.ckpt')
    
    try:
        predictor.save(model_path)
        print(f"Modele sauvegarde : {model_path}")
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde : {e}")
    
    # ETAPE 6 : Resume
    print(f"\n{'='*80}")
    print("ENTRAINEMENT TERMINE")
    print("="*80)
    print(f"Symbole :           {symbol}")
    print(f"Donnees :           {len(data)} jours")
    print(f"Epochs :            {epochs}")
    print(f"Batch size :        {batch_size}")
    print(f"Modele sauvegarde : {model_path}")
    print("="*80)
    
    return predictor


def main():
    """
    Fonction principale pour lancer l'entrainement depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(description='Entrainer un modele TFT pour le trading')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbole boursier (defaut: AAPL)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'epoques (defaut: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Taille des batchs (defaut: 64)')
    parser.add_argument('--window', type=int, default=60,
                       help='Taille de la fenetre passee (defaut: 60)')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Dossier de sauvegarde (defaut: saved_models)')
    
    args = parser.parse_args()
    
    # Je verifie que le symbole est disponible
    try:
        #  Utiliser le même chemin que train_tft_model()
        csv_path = CSV_PATH        
        
        available_symbols = get_available_symbols(csv_path)
        
        if args.symbol.upper() not in available_symbols:
            print(f"ERREUR : Symbole '{args.symbol}' non disponible")
            print(f"Symboles disponibles : {', '.join(available_symbols)}")
            return
    except Exception as e:
        print(f"Erreur lors de la verification des symboles : {e}")
        return
    
    # Je lance l'entrainement
    train_tft_model(
        symbol=args.symbol.upper(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_encoder_length=args.window,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()