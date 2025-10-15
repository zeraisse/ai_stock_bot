# AI Stock Bot

## Description

Projet d'intelligence artificielle pour le trading automatisé utilisant des données boursières historiques. Le projet comprend un environnement de trading compatible Gym pour l'entraînement d'agents d'apprentissage par renforcement, avec un agent DQN (Deep Q-Network) intégré et visualisation en temps réel.

## Structure du projet

```
ai_stock_bot/
├── stock_dataset.py           # Script de génération/formatage des données CSV
├── requirements.txt           # Dépendances Python
├── models/                    # Dossier pour les modèles sauvegardés
├── DQN/                       # Dossier des agents DQN
│   ├── dqn_agent.py          # Agent DQN avec visualisation en temps réel
│   └── dqn_trader.py         # Script principal d'entraînement DQN
├── datatset/                  # Dossier contenant les données CSV
│   └── top10_stocks_2025_clean_international.csv
└── env_trading/               # Environnement de trading
    ├── trading_env.py         # Environnement principal compatible Gym
    └── test_custom.py         # Script de test avec stratégie personnalisée
```

## Prérequis

- Python 3.8+
- GPU recommandé pour l'entraînement DQN (optionnel)
- Environnement virtuel activé

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

### Entraînement de l'agent DQN (Principal)

```bash
# Aller dans le dossier DQN
cd DQN

# Entraînement avec visualisation en temps réel (recommandé)
python dqn_trader.py --symbol AAPL --episodes 1000

# Entraînement sans visualisation (plus rapide)
python dqn_trader.py --symbol TSLA --episodes 500 --no-visual

# Autres symboles disponibles
python dqn_trader.py --symbol NVDA --episodes 2000
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

## Agent DQN

### Fonctionnalités
- **Réseau de neurones profond** avec couches cachées configurables
- **Replay buffer** pour l'apprentissage hors-politique
- **Target network** pour la stabilité d'entraînement
- **Exploration epsilon-greedy** avec décroissance adaptative
- **Visualisation en temps réel** des métriques d'entraînement
- **Sauvegarde automatique** des modèles et checkpoints

### Visualisation en temps réel
L'entraînement affiche 6 graphiques en continu :
- Récompenses par épisode (avec moyenne mobile)
- Évolution du Net Worth
- Decay de l'epsilon (exploration)
- Loss d'entraînement
- Distribution des actions prises
- Q-values moyennes

### Modèles sauvegardés
- `models/dqn_[symbol]_latest.pth` : Dernier modèle entraîné
- `models/dqn_[symbol]_ep[X]_[timestamp].pth` : Checkpoints périodiques
- `models/training_plots_[symbol]_[timestamp].png` : Graphiques de performance

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

### Observation
Vecteur de 13 dimensions comprenant :
- État du portefeuille (cash, actions, net worth)
- Données de marché actuelles (OHLCV)
- Indicateurs techniques (RSI, MACD, etc.)

## Symboles disponibles

AAPL, AMZN, GOOGL, JNJ, META, MSFT, NVDA, TSLA, UNH

## Performances typiques

L'agent DQN apprend généralement à :
- Dépasser le buy-and-hold après 200-500 épisodes
- Atteindre des ROI de 5-15% sur des périodes de test
- Réduire les pertes dans les marchés baissiers
- Optimiser le timing d'entrée/sortie

## Paramètres DQN avancés

Pour modifier les hyperparamètres, éditez `dqn_trader.py` :
```python
self.agent = DQNAgent(
    lr=0.001,           # Learning rate
    gamma=0.95,         # Discount factor
    epsilon_start=1.0,  # Exploration initiale
    epsilon_end=0.05,   # Exploration finale
    memory_size=50000,  # Taille du replay buffer
    batch_size=32,      # Taille des batches
    target_update=100   # Fréquence maj target network
)
```

## Prochaines étapes

- Entraînement d'agents plus sophistiqués (PPO, A3C)
- Intégration de modèles de prédiction (LSTM, Transformer)
- Trading multi-symboles simultané
- Backtesting sur données historiques étendues
- Déploiement en trading papier/réel