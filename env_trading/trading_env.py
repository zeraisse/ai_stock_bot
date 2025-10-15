"""
Architecture micro-services simple:
- TradingEnv: Orchestrateur principal (marché + règles)
- Portfolio: Gestion du portefeuille
- RewardCalculator: Calcul des récompenses
- ObservationBuilder: Construction des observations

Extensible pour:
- CoDeepNEAT + PPO (actions continues possibles)
- TFT + LNN (ajout facile de prédictions externes)
"""

import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import yfinance as yf


# 1. CONFIGURATION
@dataclass
class TradingConfig:
    """Configuration de l'environnement de trading."""
    
    # Paramètres financiers
    initial_balance: float = 10_000
    transaction_fee: float = 0.001  # 0.1%
    
    # Paramètres de trading
    allow_short: bool = False  # Pour futures extensions
    max_shares_per_trade: Optional[int] = None
    
    # Récompenses
    reward_type: str = "profit"  # "profit", "sharpe", "sortino"
    risk_free_rate: float = 0.02  # 2% annuel
    
    # Observation
    lookback_window: int = 60  # Fenêtre de prix passés
    include_technical_indicators: bool = True
    
    def __post_init__(self):
        """Validation de la configuration."""
        assert self.initial_balance > 0, "Balance initiale doit être > 0"
        assert 0 <= self.transaction_fee < 0.1, "Frais doivent être entre 0 et 10%"
        assert self.reward_type in ["profit", "sharpe", "sortino"], "Type de récompense invalide"


# 2. PORTFOLIO - Gestion du Portefeuille

class Portfolio:
    """
    Micro-service: Gestion du portefeuille.
    
    Responsabilités:
    - Gérer le cash et les positions
    - Exécuter les ordres avec frais
    - Calculer la valeur nette
    """
    
    def __init__(self, initial_balance: float, transaction_fee: float = 0.001):
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()
    
    def reset(self):
        """Réinitialiser le portefeuille."""
        self.cash = self.initial_balance
        self.shares_held = 0.0
        self.cost_basis = 0.0  # Prix d'achat moyen
        self.trades_history = []
    
    def get_net_worth(self, current_price: float) -> float:
        """Calculer la valeur nette actuelle."""
        return self.cash + self.shares_held * current_price
    
    def execute_buy(self, current_price: float, amount: Optional[float] = None) -> Dict:
        """
        Exécuter un achat.
        
        Args:
            current_price: Prix actuel de l'action
            amount: Montant à investir (None = tout le cash)
        
        Returns:
            Dict avec détails de la transaction
        """
        if amount is None:
            amount = self.cash
        
        # Calculer avec frais
        shares_to_buy = amount / (current_price * (1 + self.transaction_fee))
        cost = shares_to_buy * current_price * (1 + self.transaction_fee)
        
        if cost > self.cash:
            return {"success": False, "reason": "Insufficient funds"}
        
        # Exécuter l'achat
        self.cash -= cost
        self.shares_held += shares_to_buy
        
        # Mettre à jour le coût de base moyen
        self.cost_basis = (self.cost_basis * (self.shares_held - shares_to_buy) + 
                          current_price * shares_to_buy) / self.shares_held if self.shares_held > 0 else 0
        
        trade = {
            "success": True,
            "type": "BUY",
            "shares": shares_to_buy,
            "price": current_price,
            "cost": cost,
            "fee": cost - shares_to_buy * current_price
        }
        self.trades_history.append(trade)
        
        return trade
    
    def execute_sell(self, current_price: float, shares: Optional[float] = None) -> Dict:
        """
        Exécuter une vente.
        
        Args:
            current_price: Prix actuel de l'action
            shares: Nombre d'actions à vendre (None = tout)
        
        Returns:
            Dict avec détails de la transaction
        """
        if shares is None:
            shares = self.shares_held
        
        if shares > self.shares_held:
            return {"success": False, "reason": "Insufficient shares"}
        
        if shares == 0:
            return {"success": False, "reason": "No shares to sell"}
        
        # Calculer avec frais
        revenue = shares * current_price * (1 - self.transaction_fee)
        
        # Exécuter la vente
        self.cash += revenue
        self.shares_held -= shares
        
        trade = {
            "success": True,
            "type": "SELL",
            "shares": shares,
            "price": current_price,
            "revenue": revenue,
            "fee": shares * current_price - revenue,
            "profit": (current_price - self.cost_basis) * shares
        }
        self.trades_history.append(trade)
        
        return trade
    
    def get_state(self, current_price: float) -> Dict:
        """Obtenir l'état actuel du portefeuille."""
        return {
            "cash": self.cash,
            "shares": self.shares_held,
            "net_worth": self.get_net_worth(current_price),
            "position_value": self.shares_held * current_price,
            "unrealized_pnl": (current_price - self.cost_basis) * self.shares_held if self.shares_held > 0 else 0
        }


# 3. REWARD CALCULATOR - Calcul des Récompenses

class RewardCalculator:
    """
    Micro-service: Calcul des récompenses.
    
    Différents types de récompenses:
    - Profit simple: variation du net worth
    - Sharpe ratio: rendement ajusté au risque
    - Sortino ratio: pénalise seulement la volatilité négative
    """
    
    def __init__(self, reward_type: str = "profit", risk_free_rate: float = 0.02):
        self.reward_type = reward_type
        self.risk_free_rate = risk_free_rate / 252  # Taux journalier
        
        # Historique pour calculs statistiques
        self.net_worth_history = []
        self.returns_history = []
    
    def reset(self):
        """Réinitialiser l'historique."""
        self.net_worth_history = []
        self.returns_history = []
    
    def calculate_reward(self, 
                        previous_net_worth: float, 
                        current_net_worth: float,
                        action: int) -> float:
        """
        Calculer la récompense selon le type configuré.
        
        Args:
            previous_net_worth: Valeur nette précédente
            current_net_worth: Valeur nette actuelle
            action: Action prise (0=hold, 1=buy, 2=sell)
        
        Returns:
            reward: Valeur de la récompense
        """
        # Éviter division par zéro
        if previous_net_worth == 0:
            return 0.0
        
        # Calculer le rendement
        returns = (current_net_worth - previous_net_worth) / previous_net_worth
        
        # Mettre à jour l'historique
        self.net_worth_history.append(current_net_worth)
        self.returns_history.append(returns)
        
        # Calculer selon le type
        if self.reward_type == "profit":
            return self._profit_reward(returns)
        elif self.reward_type == "sharpe":
            return self._sharpe_reward()
        elif self.reward_type == "sortino":
            return self._sortino_reward()
        else:
            return returns
    
    def _profit_reward(self, returns: float) -> float:
        """Récompense basée sur le profit simple."""
        return returns
    
    def _sharpe_reward(self) -> float:
        """Récompense basée sur le ratio de Sharpe."""
        if len(self.returns_history) < 2:
            return 0.0
        
        excess_returns = np.array(self.returns_history) - self.risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe
    
    def _sortino_reward(self) -> float:
        """Récompense basée sur le ratio de Sortino."""
        if len(self.returns_history) < 2:
            return 0.0
        
        excess_returns = np.array(self.returns_history) - self.risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.mean(excess_returns)
        
        sortino = np.mean(excess_returns) / np.std(downside_returns)
        return sortino


# 4. OBSERVATION BUILDER - Construction des Observations

class ObservationBuilder:
    """
    Micro-service: Construction des observations.
    
    Responsabilités:
    - Construire le vecteur d'observation
    - Ajouter les indicateurs techniques
    - Intégrer les prédictions externes (LSTM/TFT/LNN)
    """
    
    def __init__(self, 
                 data_columns: List[str],
                 lookback_window: int = 60,
                 include_technical_indicators: bool = True):
        self.data_columns = data_columns
        self.lookback_window = lookback_window
        self.include_technical_indicators = include_technical_indicators
        
        # Calculer la taille de l'observation
        self._calculate_observation_size()
    
    def _calculate_observation_size(self) -> int:
        """Calculer la taille du vecteur d'observation."""
        size = 0
        
        # Portfolio state (3 valeurs)
        size += 3  # cash_normalized, shares_normalized, net_worth_normalized
        
        # Prix actuel (nombre de colonnes dans data)
        size += len(self.data_columns)
        
        # Indicateurs techniques (si activés)
        if self.include_technical_indicators:
            size += 5  # RSI, MACD, signal, BB_position, volume_change
        
        self.observation_size = size
        return size
    
    def build_observation(self,
                         portfolio_state: Dict,
                         current_data: pd.Series,
                         initial_balance: float,
                         external_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construire le vecteur d'observation complet.
        
        Args:
            portfolio_state: État du portefeuille
            current_data: Données de marché actuelles
            initial_balance: Balance initiale pour normalisation
            external_predictions: Prédictions LSTM/TFT/LNN (optionnel)
        
        Returns:
            observation: Vecteur d'observation normalisé
        """
        obs = []
        
        # 1. État du portefeuille (normalisé)
        obs.extend([
            portfolio_state["cash"] / initial_balance,
            portfolio_state["shares"],
            portfolio_state["net_worth"] / initial_balance
        ])
        
        # 2. Données de marché actuelles
        obs.extend(current_data.values)
        
        # 3. Indicateurs techniques
        if self.include_technical_indicators:
            obs.extend(self._calculate_technical_indicators(current_data))
        
        # 4. Prédictions externes (pour intégration future)
        if external_predictions is not None:
            obs.extend(external_predictions)
            # Mettre à jour la taille si c'est la première fois
            if len(external_predictions) > 0 and self.observation_size == self._calculate_observation_size():
                self.observation_size += len(external_predictions)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_technical_indicators(self, current_data: pd.Series) -> List[float]:
        """
        Calculer des indicateurs techniques basiques.
        
        Note: Version simplifiée. Dans un vrai système, ces indicateurs
        seraient précalculés sur toute la série.
        """
        # Pour l'instant, retourner des valeurs placeholder
        # Ces valeurs devraient être calculées en amont sur toute la série
        return [
            0.5,  # RSI normalisé (0-1)
            0.0,  # MACD
            0.0,  # Signal
            0.5,  # Bollinger Band position
            0.0   # Volume change
        ]


# 5. TRADING ENVIRONMENT - Environnement Principal

class TradingEnv(Env):
    """
    Environnement de Trading Gym.
    
    Orchestrateur principal qui coordonne:
    - Portfolio: gestion du portefeuille
    - RewardCalculator: calcul des récompenses
    - ObservationBuilder: construction des observations
    
    L'environnement représente:
    - Le marché (données historiques)
    - Votre portefeuille (cash + positions)
    - Les règles de trading (frais, contraintes)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: Optional[TradingConfig] = None,
                 external_predictions: Optional[np.ndarray] = None):
        """
        Initialiser l'environnement de trading.
        
        Args:
            data: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
            config: Configuration de l'environnement
            external_predictions: Prédictions LSTM/TFT/LNN (optionnel)
        """
        super(TradingEnv, self).__init__()
        
        # Configuration
        self.config = config or TradingConfig()
        
        # Données de marché
        self.data = data.reset_index(drop=True)
        self.external_predictions = external_predictions
        
        # Micro-services
        self.portfolio = Portfolio(
            initial_balance=self.config.initial_balance,
            transaction_fee=self.config.transaction_fee
        )
        self.reward_calculator = RewardCalculator(
            reward_type=self.config.reward_type,
            risk_free_rate=self.config.risk_free_rate
        )
        self.observation_builder = ObservationBuilder(
            data_columns=list(self.data.columns),
            lookback_window=self.config.lookback_window,
            include_technical_indicators=self.config.include_technical_indicators
        )
        
        # État de l'environnement
        self.current_step = 0
        self.done = False
        self.previous_net_worth = self.config.initial_balance
        
        # Espaces Gym
        # Action: # 0=Hold, 1=Buy, 2=Sell, 3=Short
        self.action_space = Discrete(4)
        
        # Observation: vecteur de features
        obs_size = self.observation_builder.observation_size
        if external_predictions is not None:
            obs_size += external_predictions.shape[1]
        
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """
        Réinitialiser l'environnement.
        
        Returns:
            observation: Observation initiale
        """
        self.current_step = 0
        self.done = False
        self.previous_net_worth = self.config.initial_balance
        
        # Réinitialiser les micro-services
        self.portfolio.reset()
        self.reward_calculator.reset()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécuter une action dans l'environnement.
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
        
        Returns:
            observation: Nouvelle observation
            reward: Récompense
            done: Episode terminé ?
            info: Informations additionnelles
        """
        # Obtenir le prix actuel
        current_price = self.data.iloc[self.current_step]["close"]
        
        # Exécuter l'action via le Portfolio
        trade_result = self._execute_action(action, current_price)
        
        # Calculer la nouvelle valeur nette
        current_net_worth = self.portfolio.get_net_worth(current_price)
        
        # Calculer la récompense
        reward = self.reward_calculator.calculate_reward(
            previous_net_worth=self.previous_net_worth,
            current_net_worth=current_net_worth,
            action=action
        )
        
        # Mettre à jour l'état
        self.previous_net_worth = current_net_worth
        self.current_step += 1
        
        # Vérifier si terminé
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Construire l'observation
        observation = self._get_observation()
        
        # Informations additionnelles
        info = {
            "net_worth": current_net_worth,
            "cash": self.portfolio.cash,
            "shares": self.portfolio.shares_held,
            "trade_result": trade_result,
            "step": self.current_step
        }
        
        return observation, reward, self.done, info
    
    def _execute_action(self, action: int, current_price: float) -> Dict:
        """Exécuter l'action via le Portfolio."""
        if action == 1:  # Buy
            return self.portfolio.execute_buy(current_price)
        elif action == 2:  # Sell
            return self.portfolio.execute_sell(current_price)
        elif action == 3:  # Short
            return {"success": True, "type": "SHORT", "note": "Short pas encore implémenté"}
        else:  # Hold
            return {"success": True, "type": "HOLD"}
    
    def _get_observation(self) -> np.ndarray:
        """Construire l'observation via ObservationBuilder."""
        portfolio_state = self.portfolio.get_state(
            self.data.iloc[self.current_step]["close"]
        )
        
        current_data = self.data.iloc[self.current_step]
        
        # Récupérer les prédictions externes si disponibles
        ext_pred = None
        if self.external_predictions is not None:
            ext_pred = self.external_predictions[self.current_step]
        
        return self.observation_builder.build_observation(
            portfolio_state=portfolio_state,
            current_data=current_data,
            initial_balance=self.config.initial_balance,
            external_predictions=ext_pred
        )
    
    def render(self, mode='human'):
        """Afficher l'état actuel de l'environnement."""
        current_price = self.data.iloc[self.current_step]["close"]
        portfolio_state = self.portfolio.get_state(current_price)
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{len(self.data)-1}")
        print(f"{'='*60}")
        print(f"Prix actuel:        ${current_price:.2f}")
        print(f"Cash:               ${portfolio_state['cash']:.2f}")
        print(f"Actions détenues:   {portfolio_state['shares']:.2f}")
        print(f"Valeur position:    ${portfolio_state['position_value']:.2f}")
        print(f"Net Worth:          ${portfolio_state['net_worth']:.2f}")
        print(f"P&L non réalisé:    ${portfolio_state['unrealized_pnl']:.2f}")
        print(f"ROI:                {((portfolio_state['net_worth'] - self.config.initial_balance) / self.config.initial_balance * 100):.2f}%")
        print(f"{'='*60}\n")
    
    def get_portfolio_history(self) -> List[Dict]:
        """Obtenir l'historique des trades."""
        return self.portfolio.trades_history


# 6. EXEMPLE D'UTILISATION

if __name__ == "__main__":
    # data yahoo finance 
    data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Configuration personnalisée
    config = TradingConfig(
        initial_balance=10_000,
        transaction_fee=0.001,
        reward_type="profit",
        include_technical_indicators=True
    )
    
    # Créer l'environnement
    env = TradingEnv(data=data, config=config)
    
    # Tester avec des actions aléatoires
    print("Test de l'environnement avec des actions aléatoires\n")
    
    obs = env.reset()
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Action space: {env.action_space}")
    print(f"✓ Observation space: {env.observation_space}\n")
    
    # Simuler quelques étapes
    for step in range(10):
        action = env.action_space.sample()  # Action aléatoire
        obs, reward, done, info = env.step(action)
        
        action_names = ['HOLD', 'BUY', 'SELL', 'SHORT']
        print(f"Step {step}: Action={action_names[action]}, Reward={reward:.6f}, Net Worth=${info['net_worth']:.2f}")
        
        if done:
            break
    
    print(f"\n✓ Environnement fonctionnel!")
    print(f"✓ Historique des trades: {len(env.get_portfolio_history())} transactions")