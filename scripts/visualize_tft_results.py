"""
Script de visualisation des resultats TFT vs LSTM pour le rapport au prof.

Je cree ce script pour generer tous les graphiques necessaires a la demo:
1. Comparaison predictions TFT vs LSTM vs Reelles
2. Distribution des erreurs
3. Metriques de performance (MSE, MAE, R2)
4. Attention weights (si disponible)
5. Evolution des Q-values DQN-TFT vs DQN-LSTM
6. Profits cumules des deux approches
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# J'ajoute les chemins
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'newDQN'))

from models.lstm.lstm_predictor import LSTMPredictor
from models.tft.tft_predictor import TFTPredictor

# Je configure le style des graphiques pour le prof
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def load_data(csv_path, symbol='AAPL'):
    """
    Je charge les donnees boursieres.
    """
    data = pd.read_csv(csv_path)
    data = data[data['Symbol'] == symbol].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Je renomme les colonnes en minuscules
    data.columns = data.columns.str.lower()
    
    return data


def prepare_models_and_predictions(data, lstm_model_path, tft_model_path):
    """
    Je charge les deux modeles et genere les predictions.
    """
    print("\n" + "="*80)
    print("PREPARATION DES MODELES ET PREDICTIONS")
    print("="*80)
    
    # LSTM
    print("\n1. Chargement LSTM...")
    lstm_predictor = LSTMPredictor(sequence_length=60)
    X_lstm, y_lstm = lstm_predictor.prepare_data(data)
    
    if os.path.exists(lstm_model_path):
        lstm_predictor.load(lstm_model_path)
        print(f"   LSTM charge depuis {lstm_model_path}")
    else:
        print(f"   ATTENTION: Modele LSTM introuvable, je l'entraine...")
        split = int(len(X_lstm) * 0.8)
        lstm_predictor.train(X_lstm[:split], y_lstm[:split], 
                            X_lstm[split:], y_lstm[split:], 
                            epochs=30)
    
    lstm_preds = lstm_predictor.predict(X_lstm)
    
    # TFT
    print("\n2. Chargement TFT...")
    tft_predictor = TFTPredictor(sequence_length=60)
    X_tft, y_tft = tft_predictor.prepare_data(data)
    
    if os.path.exists(tft_model_path):
        tft_predictor.load(tft_model_path)
        print(f"   TFT charge depuis {tft_model_path}")
    else:
        print(f"   ATTENTION: Modele TFT introuvable, je l'entraine...")
        split = int(len(X_tft) * 0.8)
        tft_predictor.train(X_tft[:split], y_tft[:split],
                           X_tft[split:], y_tft[split:],
                           epochs=30)
    
    tft_preds = tft_predictor.predict(X_tft)
    
    # Je m'assure que les tailles correspondent
    min_len = min(len(lstm_preds), len(tft_preds), len(y_lstm))
    
    return {
        'lstm_preds': lstm_preds[:min_len],
        'tft_preds': tft_preds[:min_len],
        'y_true': y_lstm[:min_len],
        'dates': data['date'].values[60:60+min_len],  # J'exclus les 60 premiers jours
        'lstm_predictor': lstm_predictor,
        'tft_predictor': tft_predictor
    }


def plot_predictions_comparison(results, save_path='visualizations/predictions_comparison.png'):
    """
    Je cree le graphique de comparaison des predictions.
    """
    print("\n3. Generation du graphique de comparaison...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Je prends les 200 derniers jours pour la lisibilite
    n_display = min(200, len(results['dates']))
    dates = results['dates'][-n_display:]
    
    # GRAPHIQUE 1: Prix predit
    ax1 = axes[0]
    ax1.plot(dates, results['y_true'][-n_display:, 0], 
             label='Prix Reel', color='black', linewidth=2, alpha=0.7)
    ax1.plot(dates, results['lstm_preds'][-n_display:, 0], 
             label='LSTM', color='blue', linewidth=1.5, alpha=0.6)
    ax1.plot(dates, results['tft_preds'][-n_display:, 0], 
             label='TFT', color='red', linewidth=1.5, alpha=0.6)
    ax1.set_title('Comparaison Prix Predits: TFT vs LSTM vs Reel', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prix Normalise', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # GRAPHIQUE 2: Tendance predite
    ax2 = axes[1]
    ax2.plot(dates, results['y_true'][-n_display:, 1], 
             label='Tendance Reelle', color='black', linewidth=2, alpha=0.7)
    ax2.plot(dates, results['lstm_preds'][-n_display:, 1], 
             label='LSTM', color='blue', linewidth=1.5, alpha=0.6)
    ax2.plot(dates, results['tft_preds'][-n_display:, 1], 
             label='TFT', color='red', linewidth=1.5, alpha=0.6)
    ax2.set_title('Comparaison Tendances', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Tendance (-1/0/1)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # GRAPHIQUE 3: Volatilite predite
    ax3 = axes[2]
    ax3.plot(dates, results['y_true'][-n_display:, 2], 
             label='Volatilite Reelle', color='black', linewidth=2, alpha=0.7)
    ax3.plot(dates, results['lstm_preds'][-n_display:, 2], 
             label='LSTM', color='blue', linewidth=1.5, alpha=0.6)
    ax3.plot(dates, results['tft_preds'][-n_display:, 2], 
             label='TFT', color='red', linewidth=1.5, alpha=0.6)
    ax3.set_title('Comparaison Volatilite', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Volatilite', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}")
    plt.close()


def plot_error_distributions(results, save_path='visualizations/error_distributions.png'):
    """
    Je cree les distributions des erreurs.
    """
    print("\n4. Generation des distributions d'erreurs...")
    
    # Je calcule les erreurs pour les prix (colonne 0)
    lstm_errors = results['y_true'][:, 0] - results['lstm_preds'][:, 0]
    tft_errors = results['y_true'][:, 0] - results['tft_preds'][:, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogrammes
    ax1 = axes[0]
    ax1.hist(lstm_errors, bins=50, alpha=0.6, label='LSTM', color='blue', edgecolor='black')
    ax1.hist(tft_errors, bins=50, alpha=0.6, label='TFT', color='red', edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.set_title('Distribution des Erreurs de Prediction', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Erreur (Reel - Predit)', fontsize=12)
    ax1.set_ylabel('Frequence', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Boxplots
    ax2 = axes[1]
    bp = ax2.boxplot([lstm_errors, tft_errors], 
                     labels=['LSTM', 'TFT'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax2.set_title('Boxplot des Erreurs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Erreur', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}")
    plt.close()


def plot_metrics_comparison(results, save_path='visualizations/metrics_comparison.png'):
    """
    Je cree le tableau comparatif des metriques.
    """
    print("\n5. Calcul des metriques de performance...")
    
    # Je calcule les metriques pour les prix (colonne 0)
    lstm_mse = mean_squared_error(results['y_true'][:, 0], results['lstm_preds'][:, 0])
    tft_mse = mean_squared_error(results['y_true'][:, 0], results['tft_preds'][:, 0])
    
    lstm_mae = mean_absolute_error(results['y_true'][:, 0], results['lstm_preds'][:, 0])
    tft_mae = mean_absolute_error(results['y_true'][:, 0], results['tft_preds'][:, 0])
    
    lstm_r2 = r2_score(results['y_true'][:, 0], results['lstm_preds'][:, 0])
    tft_r2 = r2_score(results['y_true'][:, 0], results['tft_preds'][:, 0])
    
    # Je cree le graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['MSE', 'MAE', 'R² Score']
    lstm_values = [lstm_mse, lstm_mae, lstm_r2]
    tft_values = [tft_mse, tft_mae, tft_r2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, tft_values, width, label='TFT', color='red', alpha=0.7)
    
    # J'ajoute les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Comparaison des Metriques: TFT vs LSTM', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valeur', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # J'ajoute un texte avec le resume
    improvement_mse = ((lstm_mse - tft_mse) / lstm_mse) * 100
    improvement_mae = ((lstm_mae - tft_mae) / lstm_mae) * 100
    
    text = f"Amelioration TFT vs LSTM:\n"
    text += f"MSE: {improvement_mse:+.2f}%\n"
    text += f"MAE: {improvement_mae:+.2f}%"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}")
    plt.close()
    
    # J'affiche aussi dans la console
    print("\n" + "="*80)
    print("METRIQUES DE PERFORMANCE")
    print("="*80)
    print(f"{'Metrique':<15} {'LSTM':<15} {'TFT':<15} {'Amelioration':<15}")
    print("-"*80)
    print(f"{'MSE':<15} {lstm_mse:<15.6f} {tft_mse:<15.6f} {improvement_mse:+.2f}%")
    print(f"{'MAE':<15} {lstm_mae:<15.6f} {tft_mae:<15.6f} {improvement_mae:+.2f}%")
    print(f"{'R² Score':<15} {lstm_r2:<15.6f} {tft_r2:<15.6f}")
    print("="*80)


def plot_architecture_comparison(save_path='visualizations/architecture_comparison.png'):
    """
    Je cree un schema explicatif de l'architecture TFT vs LSTM.
    """
    print("\n6. Generation du schema d'architecture...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # LSTM
    ax1 = axes[0]
    ax1.text(0.5, 0.9, 'LSTM ARCHITECTURE', ha='center', fontsize=14, fontweight='bold')
    
    components_lstm = [
        'Input (Prix, Volume, etc.)',
        '↓',
        'Embedding Layer',
        '↓',
        'LSTM Layer 1',
        '↓',
        'LSTM Layer 2',
        '↓',
        'Attention Mechanism',
        '↓',
        'Fully Connected',
        '↓',
        'Output (Prix, Tendance, Volatilite)'
    ]
    
    y_pos = 0.8
    for comp in components_lstm:
        ax1.text(0.5, y_pos, comp, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        y_pos -= 0.065
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # TFT
    ax2 = axes[1]
    ax2.text(0.5, 0.9, 'TFT ARCHITECTURE', ha='center', fontsize=14, fontweight='bold')
    
    components_tft = [
        'Input (Prix, Volume, etc.)',
        '↓',
        'Variable Selection Network (VSN)',
        '↓',
        'Static Covariate Encoder',
        '↓',
        'LSTM Encoder (Bidirectionnel)',
        '↓',
        'Multi-Head Attention (4 tetes)',
        '↓',
        'Gated Residual Network (GRN)',
        '↓',
        'Quantile Output Layer',
        '↓',
        'Output (Predictions + Intervalles)'
    ]
    
    y_pos = 0.8
    for comp in components_tft:
        ax2.text(0.5, y_pos, comp, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        y_pos -= 0.06
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}")
    plt.close()


def generate_summary_report(results, save_path='visualizations/summary_report.txt'):
    """
    Je genere un rapport texte resumant les resultats.
    """
    print("\n7. Generation du rapport texte...")
    
    # Je calcule toutes les metriques
    lstm_mse = mean_squared_error(results['y_true'][:, 0], results['lstm_preds'][:, 0])
    tft_mse = mean_squared_error(results['y_true'][:, 0], results['tft_preds'][:, 0])
    lstm_mae = mean_absolute_error(results['y_true'][:, 0], results['lstm_preds'][:, 0])
    tft_mae = mean_absolute_error(results['y_true'][:, 0], results['tft_preds'][:, 0])
    lstm_r2 = r2_score(results['y_true'][:, 0], results['lstm_preds'][:, 0])
    tft_r2 = r2_score(results['y_true'][:, 0], results['tft_preds'][:, 0])
    
    report = f"""
{'='*80}
RAPPORT DE COMPARAISON TFT vs LSTM
{'='*80}

Date de generation: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Nombre d'echantillons: {len(results['y_true'])}

{'='*80}
METRIQUES DE PERFORMANCE
{'='*80}

Prix (Feature principale):
  LSTM:
    - MSE: {lstm_mse:.6f}
    - MAE: {lstm_mae:.6f}
    - R²:  {lstm_r2:.6f}
  
  TFT:
    - MSE: {tft_mse:.6f}
    - MAE: {tft_mae:.6f}
    - R²:  {tft_r2:.6f}
  
  Amelioration:
    - MSE: {((lstm_mse - tft_mse) / lstm_mse * 100):+.2f}%
    - MAE: {((lstm_mae - tft_mae) / lstm_mae * 100):+.2f}%

{'='*80}
ARCHITECTURE
{'='*80}

LSTM:
  - 2 couches LSTM
  - Hidden size: 128
  - Attention simple
  - Dropout: 20%

TFT:
  - LSTM Bidirectionnel (2 couches)
  - Hidden size: 128
  - Multi-Head Attention (4 tetes)
  - Dropout: 20%
  - Variable Selection Network
  - Gated Residual Network

{'='*80}
CONCLUSION
{'='*80}

Le modele TFT montre {'une amelioration' if tft_mse < lstm_mse else 'une degradation'} 
par rapport au LSTM sur les predictions de prix.

L'architecture TFT plus complexe permet de capturer des dependances 
long-terme grace a son mecanisme d'attention multi-tete et sa capacite
de selection automatique des features importantes.

Pour le trading algorithmique, {'TFT' if tft_mse < lstm_mse else 'LSTM'} est recommande
pour les predictions de prix.

{'='*80}
"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   Sauvegarde: {save_path}")
    print("\n" + report)


def main():
    """
    Je genere tous les graphiques de comparaison.
    """
    print("\n" + "="*80)
    print("SCRIPT DE VISUALISATION TFT vs LSTM")
    print("Pour demonstration au professeur")
    print("="*80)
    
    # Configuration
    csv_path = '../dataset/top10_stocks_2025.csv'
    symbol = 'AAPL'
    lstm_model_path = 'saved_models/lstm_aapl.pth'
    tft_model_path = 'saved_models/tft_aapl.pth'
    
    # Je charge les donnees
    print("\n1. Chargement des donnees...")
    data = load_data(csv_path, symbol)
    print(f"   Donnees chargees: {len(data)} jours pour {symbol}")
    
    # Je prepare les modeles et predictions
    results = prepare_models_and_predictions(data, lstm_model_path, tft_model_path)
    
    # Je genere tous les graphiques
    print("\n" + "="*80)
    print("GENERATION DES VISUALISATIONS")
    print("="*80)
    
    plot_predictions_comparison(results)
    plot_error_distributions(results)
    plot_metrics_comparison(results)
    plot_architecture_comparison()
    generate_summary_report(results)
    
    print("\n" + "="*80)
    print("GENERATION TERMINEE")
    print("="*80)
    print("\nTous les graphiques sont dans le dossier: visualizations/")
    print("\nFichiers generes:")
    print("  1. predictions_comparison.png - Comparaison des predictions")
    print("  2. error_distributions.png - Distribution des erreurs")
    print("  3. metrics_comparison.png - Metriques de performance")
    print("  4. architecture_comparison.png - Schema des architectures")
    print("  5. summary_report.txt - Rapport texte complet")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()