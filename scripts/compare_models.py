"""
Script pour comparer les performances LSTM vs TFT.

Je cree ce script pour :
1. Charger les modeles LSTM et TFT entraines
2. Generer des predictions sur les memes donnees
3. Calculer les metriques de performance
4. Visualiser les comparaisons
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# J'ajoute les chemins vers mes modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from models.lstm import LSTMPredictor
from models.tft import TFTPredictor
from env_trading.trading_env import load_stock_data_from_csv
from preprocessing.data_processor import DataProcessor


def compare_models(symbol='AAPL',
                  lstm_model_path=None,
                  tft_model_path=None,
                  output_dir='comparison_results'):
    """
    Je compare les performances de LSTM et TFT.
    
    Args:
        symbol: Symbole boursier
        lstm_model_path: Chemin vers le modele LSTM
        tft_model_path: Chemin vers le modele TFT
        output_dir: Dossier de sortie pour les resultats
    """
    print("="*80)
    print(f"COMPARAISON LSTM VS TFT POUR {symbol}")
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
    
    # ETAPE 2 : Preparer les donnees pour LSTM
    print(f"\n2. Preparation des donnees pour LSTM...")
    print("-"*80)
    
    lstm_predictor = LSTMPredictor(sequence_length=60)
    X_lstm, y_lstm = lstm_predictor.prepare_data(data)
    
    # Je split les donnees
    processor = DataProcessor(sequence_length=60)
    X_train, y_train, X_val, y_val, X_test, y_test = processor.split_data(
        X_lstm, y_lstm,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"Donnees LSTM preparees")
    print(f"  - Train : {len(X_train)} samples")
    print(f"  - Val : {len(X_val)} samples")
    print(f"  - Test : {len(X_test)} samples")
    
    # ETAPE 3 : Charger le modele LSTM
    print(f"\n3. Chargement du modele LSTM...")
    print("-"*80)
    
    if lstm_model_path is None:
        lstm_model_path = os.path.join('saved_models', f'lstm_{symbol.lower()}.pth')
    
    if not os.path.exists(lstm_model_path):
        print(f"ERREUR : Modele LSTM non trouve : {lstm_model_path}")
        print(f"Vous devez d'abord entrainer LSTM")
        return
    
    try:
        lstm_predictor.load(lstm_model_path)
        print(f"Modele LSTM charge")
    except Exception as e:
        print(f"ERREUR lors du chargement LSTM : {e}")
        return
    
    # ETAPE 4 : Charger le modele TFT
    print(f"\n4. Chargement du modele TFT...")
    print("-"*80)
    
    if tft_model_path is None:
        tft_model_path = os.path.join('saved_models', f'tft_{symbol.lower()}.ckpt')
    
    if not os.path.exists(tft_model_path):
        print(f"ERREUR : Modele TFT non trouve : {tft_model_path}")
        print(f"Vous devez d'abord entrainer TFT")
        return
    
    tft_predictor = TFTPredictor(max_encoder_length=60)
    X_tft, y_tft = tft_predictor.prepare_data(data)
    
    try:
        tft_predictor.load(tft_model_path)
        print(f"Modele TFT charge")
    except Exception as e:
        print(f"ERREUR lors du chargement TFT : {e}")
        return
    
    # ETAPE 5 : Generer les predictions
    print(f"\n5. Generation des predictions...")
    print("-"*80)
    
    print("Predictions LSTM...")
    lstm_predictions = lstm_predictor.predict(X_test)
    
    print("Predictions TFT...")
    tft_predictions = tft_predictor.predict(X_tft)
    # Je prends seulement la partie test pour TFT
    tft_predictions_test = tft_predictions[-len(X_test):]
    
    print(f"Predictions generees")
    print(f"  - LSTM : {lstm_predictions.shape}")
    print(f"  - TFT : {tft_predictions_test.shape}")
    
    # ETAPE 6 : Calculer les metriques
    print(f"\n6. Calcul des metriques...")
    print("-"*80)
    
    # Je calcule les metriques pour LSTM
    lstm_metrics = calculate_metrics(y_test, lstm_predictions)
    
    # Je calcule les metriques pour TFT
    tft_metrics = calculate_metrics(y_test, tft_predictions_test)
    
    # J'affiche les resultats
    print("\nMETRIQUES LSTM :")
    print("-"*40)
    for key, value in lstm_metrics.items():
        print(f"  {key:20s} : {value:.6f}")
    
    print("\nMETRIQUES TFT :")
    print("-"*40)
    for key, value in tft_metrics.items():
        print(f"  {key:20s} : {value:.6f}")
    
    # Je calcule l'amelioration
    print("\nAMELIORATION TFT vs LSTM :")
    print("-"*40)
    for key in lstm_metrics.keys():
        if key.endswith('mae') or key.endswith('mse') or key.endswith('rmse'):
            # Pour ces metriques, plus bas = mieux
            improvement = ((lstm_metrics[key] - tft_metrics[key]) / lstm_metrics[key]) * 100
            if improvement > 0:
                print(f"  {key:20s} : {improvement:+.2f}% (mieux)")
            else:
                print(f"  {key:20s} : {improvement:+.2f}% (moins bien)")
    
    # ETAPE 7 : Visualiser les resultats
    print(f"\n7. Visualisation des resultats...")
    print("-"*80)
    
    # Je cree le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Je visualise les predictions
    visualize_predictions(
        y_test, 
        lstm_predictions, 
        tft_predictions_test,
        output_dir,
        symbol
    )
    
    # Je visualise les metriques
    visualize_metrics(
        lstm_metrics,
        tft_metrics,
        output_dir,
        symbol
    )
    
    # ETAPE 8 : Sauvegarder les resultats
    print(f"\n8. Sauvegarde des resultats...")
    print("-"*80)
    
    results_path = os.path.join(output_dir, f'comparison_{symbol.lower()}.csv')
    results_df = pd.DataFrame({
        'Metric': list(lstm_metrics.keys()),
        'LSTM': list(lstm_metrics.values()),
        'TFT': list(tft_metrics.values())
    })
    results_df.to_csv(results_path, index=False)
    print(f"Resultats sauvegardes : {results_path}")
    
    # ETAPE 9 : Resume
    print(f"\n{'='*80}")
    print("COMPARAISON TERMINEE")
    print("="*80)
    print(f"Symbole :            {symbol}")
    print(f"Test samples :       {len(X_test)}")
    print(f"Resultats :          {results_path}")
    print(f"Graphiques :         {output_dir}")
    print("="*80)
    
    return lstm_metrics, tft_metrics


def calculate_metrics(y_true, y_pred):
    """
    Je calcule les metriques de performance.
    
    Args:
        y_true: Valeurs reelles (n_samples, 3)
        y_pred: Predictions (n_samples, 3)
        
    Returns:
        metrics: Dictionnaire avec les metriques
    """
    # Je calcule les metriques globales
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    # Je calcule les metriques par feature
    feature_names = ['prix', 'tendance', 'volatilite']
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }
    
    for i, name in enumerate(feature_names):
        metrics[f'{name}_mae'] = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
        metrics[f'{name}_mse'] = np.mean((y_pred[:, i] - y_true[:, i]) ** 2)
    
    return metrics


def visualize_predictions(y_true, lstm_pred, tft_pred, output_dir, symbol):
    """
    Je visualise les predictions LSTM vs TFT.
    
    Args:
        y_true: Valeurs reelles
        lstm_pred: Predictions LSTM
        tft_pred: Predictions TFT
        output_dir: Dossier de sortie
        symbol: Symbole boursier
    """
    feature_names = ['Prix Ratio', 'Tendance', 'Volatilite']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Comparaison LSTM vs TFT - {symbol}', fontsize=16)
    
    # Je limite a 200 points pour la lisibilite
    n_points = min(200, len(y_true))
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(y_true[:n_points, i], label='Reel', color='black', linewidth=2)
        ax.plot(lstm_pred[:n_points, i], label='LSTM', color='blue', alpha=0.7)
        ax.plot(tft_pred[:n_points, i], label='TFT', color='red', alpha=0.7)
        
        ax.set_title(name)
        ax.set_xlabel('Temps')
        ax.set_ylabel('Valeur')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'predictions_{symbol.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegarde : {output_path}")
    plt.close()


def visualize_metrics(lstm_metrics, tft_metrics, output_dir, symbol):
    """
    Je visualise les metriques de comparaison.
    
    Args:
        lstm_metrics: Metriques LSTM
        tft_metrics: Metriques TFT
        output_dir: Dossier de sortie
        symbol: Symbole boursier
    """
    # Je selectionne les metriques importantes
    metrics_to_plot = ['mae', 'rmse', 'prix_mae', 'tendance_mae', 'volatilite_mae']
    
    lstm_values = [lstm_metrics[m] for m in metrics_to_plot]
    tft_values = [tft_metrics[m] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, tft_values, width, label='TFT', color='red', alpha=0.7)
    
    ax.set_xlabel('Metriques')
    ax.set_ylabel('Valeur')
    ax.set_title(f'Comparaison des Metriques - {symbol}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'metrics_{symbol.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegarde : {output_path}")
    plt.close()


def main():
    """
    Fonction principale pour lancer la comparaison depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(description='Comparer LSTM et TFT')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbole boursier (defaut: AAPL)')
    parser.add_argument('--lstm', type=str, default=None,
                       help='Chemin vers le modele LSTM')
    parser.add_argument('--tft', type=str, default=None,
                       help='Chemin vers le modele TFT')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='Dossier de sortie')
    
    args = parser.parse_args()
    
    # Je lance la comparaison
    compare_models(
        symbol=args.symbol.upper(),
        lstm_model_path=args.lstm,
        tft_model_path=args.tft,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()