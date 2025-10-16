"""
Script pour generer des predictions TFT pour le DQN.

Je cree ce script pour :
1. Charger un modele TFT entraine
2. Generer des predictions sur les donnees
3. Sauvegarder les predictions pour le DQN
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
from env_trading.trading_env import load_stock_data_from_csv


def generate_predictions(symbol='AAPL',
                        model_path=None,
                        output_dir='predictions'):
    """
    Je genere des predictions TFT pour un symbole boursier.
    
    Args:
        symbol: Symbole boursier
        model_path: Chemin vers le modele entraine
        output_dir: Dossier de sortie pour les predictions
    """
    print("="*80)
    print(f"GENERATION DE PREDICTIONS TFT POUR {symbol}")
    print("="*80)
    
    # ETAPE 1 : Charger les donnees
    print(f"\n1. Chargement des donnees pour {symbol}...")
    print("-"*80)
    
    csv_path = os.path.join('..', '..', 'datatset', 'top10_stocks_2025_clean_international.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERREUR : Fichier non trouve : {csv_path}")
        return
    
    try:
        data = load_stock_data_from_csv(csv_path, symbol=symbol)
        print(f"Donnees chargees : {len(data)} jours")
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return
    
    # ETAPE 2 : Charger le modele TFT
    print(f"\n2. Chargement du modele TFT...")
    print("-"*80)
    
    # Je determine le chemin du modele
    if model_path is None:
        model_path = os.path.join('saved_models', f'tft_{symbol.lower()}.ckpt')
    
    if not os.path.exists(model_path):
        print(f"ERREUR : Modele non trouve : {model_path}")
        print(f"Vous devez d'abord entrainer le modele avec train_tft.py")
        return
    
    # Je cree le predicteur
    predictor = TFTPredictor(
        max_encoder_length=60,
        max_prediction_length=1,
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4
    )
    
    # Je prepare les donnees (necessaire avant load pour TFT)
    print("Preparation des donnees...")
    X, y = predictor.prepare_data(data)
    
    # Je charge le modele
    try:
        predictor.load(model_path)
        print(f"Modele charge : {model_path}")
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return
    
    # ETAPE 3 : Generer les predictions
    print(f"\n3. Generation des predictions...")
    print("-"*80)
    
    try:
        predictions = predictor.predict(X)
        print(f"Predictions generees : {predictions.shape}")
        print(f"Format : (n_samples, 3) = (n_samples, [prix_ratio, tendance, volatilite])")
        
    except Exception as e:
        print(f"ERREUR lors de la prediction : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ETAPE 4 : Analyser les predictions
    print(f"\n4. Analyse des predictions...")
    print("-"*80)
    
    print(f"Prix ratio :")
    print(f"  - Min : {predictions[:, 0].min():.4f}")
    print(f"  - Max : {predictions[:, 0].max():.4f}")
    print(f"  - Mean : {predictions[:, 0].mean():.4f}")
    
    print(f"\nTendance :")
    print(f"  - Min : {predictions[:, 1].min():.4f}")
    print(f"  - Max : {predictions[:, 1].max():.4f}")
    print(f"  - Mean : {predictions[:, 1].mean():.4f}")
    
    print(f"\nVolatilite :")
    print(f"  - Min : {predictions[:, 2].min():.4f}")
    print(f"  - Max : {predictions[:, 2].max():.4f}")
    print(f"  - Mean : {predictions[:, 2].mean():.4f}")
    
    # ETAPE 5 : Sauvegarder les predictions
    print(f"\n5. Sauvegarde des predictions...")
    print("-"*80)
    
    # Je cree le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Je sauvegarde en format numpy
    output_path = os.path.join(output_dir, f'tft_predictions_{symbol.lower()}.npy')
    np.save(output_path, predictions)
    print(f"Predictions sauvegardees : {output_path}")
    
    # Je sauvegarde aussi en CSV pour analyse
    csv_output_path = os.path.join(output_dir, f'tft_predictions_{symbol.lower()}.csv')
    pred_df = pd.DataFrame(
        predictions,
        columns=['prix_ratio', 'tendance', 'volatilite']
    )
    pred_df.to_csv(csv_output_path, index=False)
    print(f"Predictions CSV sauvegardees : {csv_output_path}")
    
    # ETAPE 6 : Instructions pour utilisation
    print(f"\n{'='*80}")
    print("PREDICTIONS GENEREES AVEC SUCCES")
    print("="*80)
    print(f"Symbole :          {symbol}")
    print(f"Predictions :      {len(predictions)} jours")
    print(f"Fichier numpy :    {output_path}")
    print(f"Fichier CSV :      {csv_output_path}")
    print("\nPour utiliser avec le DQN :")
    print(f"  predictions = np.load('{output_path}')")
    print(f"  env = TradingEnv(data, external_predictions=predictions)")
    print("="*80)
    
    return predictions


def main():
    """
    Fonction principale pour lancer la generation depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(description='Generer des predictions TFT')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbole boursier (defaut: AAPL)')
    parser.add_argument('--model', type=str, default=None,
                       help='Chemin vers le modele (defaut: saved_models/tft_SYMBOL.ckpt)')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Dossier de sortie (defaut: predictions)')
    
    args = parser.parse_args()
    
    # Je lance la generation
    generate_predictions(
        symbol=args.symbol.upper(),
        model_path=args.model,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()