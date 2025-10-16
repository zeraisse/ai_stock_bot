# Agent DQN pour Trading

## Fichiers

- `dqn_agent.py` : Classes DQN, réseau de neurones, replay buffer et visualiseur
- `dqn_trader.py` : Script principal d'entraînement
- `models/` : Modèles sauvegardés et checkpoints

## Utilisation

```bash
# Entraînement basique
python dqn_trader.py --symbol AAPL --episodes 500

# Entraînement sans visualisation
python dqn_trader.py --symbol TSLA --episodes 1000 --no-visual

# Options disponibles
python dqn_trader.py --help
```

## Arguments

- `--symbol` : Symbole boursier (AAPL, TSLA, NVDA, etc.)
- `--episodes` : Nombre d'épisodes d'entraînement
- `--no-visual` : Désactiver la visualisation en temps réel

## Modèles sauvegardés

- `models/dqn_[symbol]_latest.pth` : Dernier modèle entraîné
- `models/dqn_[symbol]_ep[X]_[timestamp].pth` : Checkpoints
- `models/training_plots_[symbol]_[timestamp].png` : Graphiques