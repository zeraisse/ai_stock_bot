"""
Script d'entraînement du modèle LSTM.

Usage:
    python train_lstm.py --symbol AAPL --epochs 100 --batch_size 32
"""

import sys
sys.path.append('..')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.lstm.lstm_predictor import LSTMPredictor
from preprocessing import DataProcessor, TechnicalIndicators
from env_trading.trading_env import load_stock_data_from_csv


def parse_args():
    """Parser les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description='Entraîner le modèle LSTM')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbole de l\'action à entraîner (default: AAPL)')
    parser.add_argument('--csv_path', type=str, 
                       default='../datatset/top10_stocks_2025_clean_international.csv',
                       help='Chemin vers le fichier CSV')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Longueur de la séquence temporelle (default: 60)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Taille des couches cachées (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Nombre de couches LSTM (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Taux de dropout (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Nombre d\'époques (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille des batchs (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Taux d\'apprentissage (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device à utiliser (cuda/cpu, default: cuda)')
    parser.add_argument('--output_dir', type=str, default='../saved_models',
                       help='Dossier de sortie pour les modèles')
    
    return parser.parse_args()


def plot_training_history(history: dict, symbol: str, output_dir: str):
    """
    Visualiser l'historique d'entraînement.
    
    Args:
        history: Historique d'entraînement
        symbol: Symbole de l'action
        output_dir: Dossier de sortie
    """
    plt.figure(figsize=(12, 5))
    
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'Courbe d\'Apprentissage - {symbol}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder
    plot_path = os.path.join(output_dir, f'training_curve_{symbol}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Graphique sauvegardé: {plot_path}")


def main():
    """Fonction principale d'entraînement."""
    # Parser les arguments
    args = parse_args()
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE LSTM")
    print("="*80)
    print(f"Symbole:          {args.symbol}")
    print(f"Séquence length:  {args.sequence_length}")
    print(f"Hidden size:      {args.hidden_size}")
    print(f"Num layers:       {args.num_layers}")
    print(f"Dropout:          {args.dropout}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Device:           {args.device}")
    print("="*80 + "\n")
    
    # ========== ÉTAPE 1: CHARGER LES DONNÉES ==========
    print("[1/5] Chargement des données...")
    try:
        data = load_stock_data_from_csv(args.csv_path, symbol=args.symbol)
        print(f"✓ {len(data)} jours de données chargés\n")
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return
    
    # ========== ÉTAPE 2: CRÉER LE PRÉDICTEUR ==========
    print("[2/5] Création du prédicteur LSTM...")
    predictor = LSTMPredictor(
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device
    )
    print("✓ Prédicteur créé\n")
    
    # ========== ÉTAPE 3: PRÉPARER LES DONNÉES ==========
    print("[3/5] Préparation des données...")
    X, y = predictor.prepare_data(data)
    
    print(f"✓ Données préparées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}\n")
    
    # Split train/val/test
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"✓ Split effectué:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples\n")
    
    # ========== ÉTAPE 4: ENTRAÎNER LE MODÈLE ==========
    print("[4/5] Entraînement du modèle...")
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # ========== ÉTAPE 5: ÉVALUER ET SAUVEGARDER ==========
    print("[5/5] Évaluation et sauvegarde...")
    
    # Évaluation sur test set
    metrics = predictor.evaluate(X_test, y_test)
    
    print(f"\n✓ Métriques sur test set:")
    print(f"  MSE:        {metrics['mse']:.6f}")
    print(f"  MAE:        {metrics['mae']:.6f}")
    print(f"  RMSE:       {metrics['rmse']:.6f}")
    print(f"  Price MAE:  {metrics.get('price_mae', 0):.6f}")
    print(f"  Trend MAE:  {metrics.get('trend_mae', 0):.6f}")
    print(f"  Vol MAE:    {metrics.get('volatility_mae', 0):.6f}")
    
    # Sauvegarder le modèle
    model_path = os.path.join(args.output_dir, f'lstm_{args.symbol}.pth')
    predictor.save(model_path)
    
    # Visualiser l'historique
    plot_training_history(history, args.symbol, args.output_dir)
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(args.output_dir, f'metrics_{args.symbol}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"MÉTRIQUES D'ÉVALUATION - {args.symbol}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Sequence length: {args.sequence_length}\n")
        f.write(f"  Hidden size:     {args.hidden_size}\n")
        f.write(f"  Num layers:      {args.num_layers}\n")
        f.write(f"  Dropout:         {args.dropout}\n")
        f.write(f"  Epochs:          {args.epochs}\n")
        f.write(f"  Batch size:      {args.batch_size}\n")
        f.write(f"  Learning rate:   {args.learning_rate}\n\n")
        f.write(f"Test Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"✓ Métriques sauvegardées: {metrics_path}")
    
    print("\n" + "="*80)
    print("✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*80)
    print(f"\nModèle sauvegardé: {model_path}")
    print(f"\nPour générer les prédictions, lancez:")
    print(f"  python generate_predictions.py --symbol {args.symbol}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()