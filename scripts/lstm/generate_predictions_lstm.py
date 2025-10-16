"""
Script de génération des prédictions LSTM.

Usage:
    python generate_predictions.py --symbol AAPL
"""

import sys

from models.lstm.lstm_predictor import LSTMPredictor
sys.path.append('..')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from env_trading.trading_env import load_stock_data_from_csv, TradingEnv, TradingConfig


def parse_args():
    """Parser les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description='Générer les prédictions LSTM')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbole de l\'action (default: AAPL)')
    parser.add_argument('--csv_path', type=str,
                       default='../datatset/top10_stocks_2025_clean_international.csv',
                       help='Chemin vers le fichier CSV')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Chemin vers le modèle (default: ../saved_models/lstm_SYMBOL.pth)')
    parser.add_argument('--output_dir', type=str, default='../saved_models',
                       help='Dossier de sortie')
    parser.add_argument('--test_env', action='store_true',
                       help='Tester l\'intégration avec TradingEnv')
    
    return parser.parse_args()


def visualize_predictions(data, predictions, symbol, output_dir):
    """
    Visualiser les prédictions.
    
    Args:
        data: DataFrame avec données originales
        predictions: Array de prédictions
        symbol: Symbole de l'action
        output_dir: Dossier de sortie
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Ajuster les longueurs
    min_len = min(len(data), len(predictions))
    data = data.iloc[:min_len]
    predictions = predictions[:min_len]
    
    # 1. Prix ratio
    axes[0].plot(predictions[:, 0], label='Prix Ratio Prédit', alpha=0.7)
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    axes[0].set_title(f'Prédictions LSTM - {symbol}: Prix Ratio', fontweight='bold')
    axes[0].set_ylabel('Prix Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Tendance
    axes[1].plot(predictions[:, 1], label='Tendance Prédite', color='green', alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('Tendance (-1=Baisse, 0=Stable, +1=Hausse)', fontweight='bold')
    axes[1].set_ylabel('Tendance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Volatilité
    axes[2].plot(predictions[:, 2], label='Volatilité Prédite', color='orange', alpha=0.7)
    axes[2].set_title('Volatilité Prédite', fontweight='bold')
    axes[2].set_ylabel('Volatilité')
    axes[2].set_xlabel('Pas de temps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_path = os.path.join(output_dir, f'predictions_{symbol}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualisation sauvegardée: {plot_path}")


def main():
    """Fonction principale de génération."""
    args = parse_args()
    
    # Déterminer le chemin du modèle
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, f'lstm_{args.symbol}.pth')
    
    print("\n" + "="*80)
    print("GÉNÉRATION DES PRÉDICTIONS LSTM")
    print("="*80)
    print(f"Symbole:     {args.symbol}")
    print(f"Modèle:      {args.model_path}")
    print(f"Output dir:  {args.output_dir}")
    print("="*80 + "\n")
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model_path):
        print(f"❌ Erreur: Modèle non trouvé: {args.model_path}")
        print(f"\nEntraînez d'abord le modèle avec:")
        print(f"  python train_lstm.py --symbol {args.symbol}")
        return
    
    # ========== ÉTAPE 1: CHARGER LES DONNÉES ==========
    print("[1/4] Chargement des données...")
    try:
        data = load_stock_data_from_csv(args.csv_path, symbol=args.symbol)
        print(f"✓ {len(data)} jours de données chargés\n")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # ========== ÉTAPE 2: CHARGER LE MODÈLE ==========
    print("[2/4] Chargement du modèle LSTM...")
    predictor = LSTMPredictor()
    predictor.load(args.model_path)
    print("✓ Modèle chargé\n")
    
    # ========== ÉTAPE 3: GÉNÉRER LES PRÉDICTIONS ==========
    print("[3/4] Génération des prédictions...")
    X, y = predictor.prepare_data(data)
    predictions = predictor.predict(X)
    
    print(f"✓ Prédictions générées: {predictions.shape}\n")
    
    # Ajouter padding au début
    padding = np.zeros((predictor.sequence_length + 1, predictions.shape[1]))
    padded_predictions = np.vstack([padding, predictions])
    
    print(f"✓ Padding ajouté: {padded_predictions.shape}")
    print(f"  Données originales: {len(data)} lignes")
    print(f"  Prédictions:        {len(padded_predictions)} lignes\n")
    
    # Sauvegarder les prédictions
    predictions_path = os.path.join(args.output_dir, f'predictions_{args.symbol}.npy')
    np.save(predictions_path, padded_predictions)
    print(f"✓ Prédictions sauvegardées: {predictions_path}\n")
    
    # ========== ÉTAPE 4: VISUALISER ==========
    print("[4/4] Visualisation des prédictions...")
    visualize_predictions(data, padded_predictions, args.symbol, args.output_dir)
    
    # Statistiques
    print(f"\n📊 STATISTIQUES DES PRÉDICTIONS:")
    print(f"  Prix ratio   - Mean: {predictions[:, 0].mean():.4f}, Std: {predictions[:, 0].std():.4f}")
    print(f"  Tendance     - Mean: {predictions[:, 1].mean():.4f}, Std: {predictions[:, 1].std():.4f}")
    print(f"  Volatilité   - Mean: {predictions[:, 2].mean():.4f}, Std: {predictions[:, 2].std():.4f}")
    
    # ========== TEST INTÉGRATION (optionnel) ==========
    if args.test_env:
        print("\n" + "="*80)
        print("TEST D'INTÉGRATION AVEC TRADINGENV")
        print("="*80 + "\n")
        
        # Charger les prédictions
        lstm_preds = np.load(predictions_path)
        
        # Ajuster les longueurs
        min_len = min(len(data), len(lstm_preds))
        data_adjusted = data.iloc[:min_len]
        lstm_preds_adjusted = lstm_preds[:min_len]
        
        # Créer l'environnement
        config = TradingConfig(initial_balance=10_000)
        env = TradingEnv(
            data=data_adjusted,
            config=config,
            external_predictions=lstm_preds_adjusted
        )
        
        print(f"✓ Environnement créé")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Les {lstm_preds_adjusted.shape[1]} features LSTM sont intégrées!")
        
        # Test rapide
        obs = env.reset()
        print(f"\n✓ Test de l'environnement:")
        print(f"  Observation shape: {obs.shape}")
        
        # Simuler 5 étapes
        print(f"\n✓ Simulation de 5 étapes...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            actions = ['HOLD', 'BUY', 'SELL', 'SHORT']
            print(f"  Step {i+1}: {actions[action]:<6} | Reward: {reward:8.6f} | Net Worth: ${info['net_worth']:,.2f}")
        
        print("\n✓ Intégration fonctionnelle!")
    
    print("\n" + "="*80)
    print("✓ GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
    print("="*80)
    print(f"\nPrédictions disponibles: {predictions_path}")
    print(f"\nVos collègues DQN peuvent maintenant utiliser:")
    print(f"  predictions = np.load('{predictions_path}')")
    print(f"  env = TradingEnv(data, external_predictions=predictions)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()