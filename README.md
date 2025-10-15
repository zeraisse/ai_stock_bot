# AI Stock Bot

## Description

Projet d'intelligence artificielle pour le trading automatisé utilisant des données boursières historiques. Le projet comprend un environnement de trading compatible Gym pour l'entraînement d'agents d'apprentissage par renforcement.

## Structure du projet

```
ai_stock_bot/
├── stock_dataset.py           # Script de génération/formatage des données CSV
├── requirements.txt           # Dépendances Python
├── datatset/                  # Dossier contenant les données CSV
│   └── top10_stocks_2025_clean_international.csv
└── env_trading/               # Environnement de trading
    ├── trading_env.py         # Environnement principal compatible Gym
    └── test_custom.py         # Script de test avec stratégie personnalisée
```

## Prérequis

- Python 3.x
- Environnement virtuel activé
- Packages requis : pandas, numpy, gym, yfinance

## Installation

1. Clonez le repository
2. Activez votre environnement virtuel
3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Génération des données

```bash
python stock_dataset.py
```

### Test de l'environnement de trading

```bash
cd env_trading
```

#### Test simple avec le script par défaut
```bash
python trading_env.py
```

#### Test avec stratégie personnalisée
```bash
python test_custom.py
```

## Format des données CSV

Le fichier CSV contient les colonnes suivantes :
- Symbol : Symbole boursier (AAPL, TSLA, NVDA, etc.)
- Date : Date au format YYYY-MM-DD
- Open : Prix d'ouverture
- High : Prix le plus haut
- Low : Prix le plus bas
- Close : Prix de clôture
- Volume : Volume des transactions

## Environnement de Trading

### Actions disponibles
- 0 : HOLD (conserver la position)
- 1 : BUY (acheter)
- 2 : SELL (vendre)
- 3 : SHORT (vente à découvert)

### Configuration
L'environnement peut être configuré via la classe `TradingConfig` :
- Balance initiale (défaut: 10,000$)
- Frais de transaction (défaut: 0.1%)
- Type de récompense (profit, sharpe, sortino)
- Paramètres d'observation

### Sortie
L'environnement retourne :
- Observation : Vecteur d'état du marché et du portefeuille
- Reward : Récompense basée sur la performance
- Done : Indicateur de fin d'épisode
- Info : Informations additionnelles (net worth, cash, shares, etc.)

## Symboles disponibles

AAPL, AMZN, GOOGL, JNJ, META, MSFT, NVDA, TSLA, UNH

## Prochaines étapes

- Entraînement d'agents d'apprentissage par renforcement (PPO, DQN)
- Optimisation des stratégies de trading
- Analyse des performances sur différents symboles
- Intégration de modèles de prédiction (LSTM, TFT)