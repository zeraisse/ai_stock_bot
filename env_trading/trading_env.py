import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import os
import yfinance as yf  # Ajout pour WebSocket
import queue  # Ajout pour gérer les mises à jour live
import threading  # Ajout pour lancer le WebSocket en background
import time  # Ajout pour fallback de polling

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
    
    # Ajout pour mode live
    max_steps_live: int = 1000  # Limite d'étapes en mode live (pour éviter boucle infinie)
    live_initial_timeout_seconds: int = 30  # Timeout initial pour premier tick
    live_step_timeout_seconds: int = 65  # Timeout d'attente de tick par step (aligné 1m)
    ws_silence_seconds_to_fallback: int = 45  # Silence WS avant fallback polling
    polling_interval_seconds: int = 2  # Intervalle polling yfinance

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
    - Le marché (données historiques ou live via WebSocket)
    - Votre portefeuille (cash + positions)
    - Les règles de trading (frais, contraintes)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 data: Optional[pd.DataFrame] = None,  # Optionnel si live=True
                 config: Optional[TradingConfig] = None,
                 external_predictions: Optional[np.ndarray] = None,
                 live: bool = False,  # Mode live
                 symbol: Optional[str] = None):  # Symbole pour live
        """
        Initialiser l'environnement de trading.
        
        Args:
            data: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume'] (requis si live=False)
            config: Configuration de l'environnement
            external_predictions: Prédictions LSTM/TFT/LNN (optionnel)
            live: Si True, utilise WebSocket Yahoo Finance au lieu de data historique
            symbol: Symbole (ex: "AAPL" ou "BTC-USD") requis en mode live
        """
        super(TradingEnv, self).__init__()
        
        # Configuration
        self.config = config or TradingConfig()
        
        # Mode live
        self.live = live
        self.symbol = symbol.upper() if symbol else None
        
        if self.live:
            if self.symbol is None:
                raise ValueError("Symbole requis en mode live (ex: 'AAPL' ou 'BTC-USD')")
            # Initialiser données live vides
            self.current_data = pd.Series({
                'open': 0.0,
                'high': 0.0,
                'low': 0.0,
                'close': 0.0,
                'volume': 0.0
            }, dtype=float)
            self.data_queue = queue.Queue()  # Pour recevoir mises à jour WebSocket
            self.ws = None
            self.ws_thread = None
        else:
            if data is None:
                raise ValueError("DataFrame 'data' requis en mode historique")
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
            data_columns=['open', 'high', 'low', 'close', 'volume'],  # Fixé pour compatibilité live
            lookback_window=self.config.lookback_window,
            include_technical_indicators=self.config.include_technical_indicators
        )
        
        # État de l'environnement
        self.current_step = 0
        self.done = False
        self.previous_net_worth = self.config.initial_balance
        
        # Espaces Gym
        self.action_space = Discrete(4)  # 0=Hold, 1=Buy, 2=Sell, 3=Short
        
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
        
        self.portfolio.reset()
        self.reward_calculator.reset()
        
        if self.live:
            self._last_ws_tick_ts = time.time()

            def message_handler(message):
                # Handler générique pour messages Yahoo non officiels
                try:
                    if isinstance(message, dict):
                        # Essaye différents formats de clé
                        if self.symbol in message:
                            msg = message[self.symbol]
                        else:
                            msg = message
                        close_val = float(msg.get('price', msg.get('close', self.current_data['close'])))
                        high_val = float(msg.get('day_high', msg.get('dayHigh', self.current_data['high'])))
                        low_val = float(msg.get('day_low', msg.get('dayLow', self.current_data['low'])))
                        open_val = float(msg.get('open_price', msg.get('open', self.current_data['open'])))
                        # volume peut être grand; caster en float pour la Series float
                        vol_raw = msg.get('day_volume', msg.get('dayVolume', self.current_data['volume']))
                        vol_val = float(vol_raw) if vol_raw is not None else float(self.current_data['volume'])
                        self.current_data['close'] = close_val
                        self.current_data['high'] = high_val
                        self.current_data['low'] = low_val
                        self.current_data['open'] = open_val
                        self.current_data['volume'] = vol_val
                        self._last_ws_tick_ts = time.time()
                        self.data_queue.put(self.current_data.copy())
                except Exception:
                    # Ignorer les messages malformés
                    pass

            # Tente WebSocket yfinance (non officiel) puis fallback polling si pas de tick
            try:
                self.ws = yf.WebSocket()
                self.ws.subscribe([self.symbol])
                self.ws_thread = threading.Thread(target=self.ws.listen, args=(message_handler,))
                self.ws_thread.daemon = True
                self.ws_thread.start()
            except Exception:
                # Si création WS échoue, démarrer directement le polling
                self._start_polling_fallback()

            # Thread de surveillance du silence WS -> fallback polling
            def _ws_silence_monitor():
                while True:
                    try:
                        time.sleep(1)
                        if time.time() - getattr(self, '_last_ws_tick_ts', 0) > self.config.ws_silence_seconds_to_fallback:
                            self._start_polling_fallback()
                            break
                    except Exception:
                        break

            mon = threading.Thread(target=_ws_silence_monitor)
            mon.daemon = True
            mon.start()

            print(f"Attente de la première mise à jour live pour {self.symbol}...")
            try:
                self.current_data = self.data_queue.get(timeout=self.config.live_initial_timeout_seconds)
                print(f"✅ Source live active. Prix initial: {self.current_data['close']:.2f}")
            except queue.Empty:
                print("⏰ Aucun tick reçu en 30s via WebSocket. Activation du fallback polling yfinance (1m)...")
                self._start_polling_fallback()
                try:
                    self.current_data = self.data_queue.get(timeout=self.config.live_initial_timeout_seconds)
                    print(f"✅ Fallback polling actif. Prix initial: {self.current_data['close']:.2f}")
                except queue.Empty:
                    print("❌ Échec du fallback polling.")
                    raise TimeoutError("Pas de données live - réseau ou source indisponible.")
        
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
        if self.live:
            try:
                self.current_data = self.data_queue.get(timeout=self.config.live_step_timeout_seconds)
                current_price = self.current_data["close"]
            except queue.Empty:
                print(f"⏰ Pas de tick reçu en {self.config.live_step_timeout_seconds}s. Utilisation de la dernière valeur.")
                current_price = self.current_data["close"]
        else:
            current_price = self.data.iloc[self.current_step]["close"]
        
        trade_result = self._execute_action(action, current_price)
        current_net_worth = self.portfolio.get_net_worth(current_price)
        reward = self.reward_calculator.calculate_reward(
            previous_net_worth=self.previous_net_worth,
            current_net_worth=current_net_worth,
            action=action
        )
        
        self.previous_net_worth = current_net_worth
        self.current_step += 1
        
        if self.live:
            self.done = self.current_step >= self.config.max_steps_live
        else:
            self.done = self.current_step >= len(self.data) - 1
        
        observation = self._get_observation()
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
        if self.live:
            current_data = self.current_data
            current_price = current_data["close"]
        else:
            current_data = self.data.iloc[self.current_step]
            current_price = current_data["close"]
        
        portfolio_state = self.portfolio.get_state(current_price)
        ext_pred = self.external_predictions[self.current_step] if self.external_predictions is not None else None
        
        return self.observation_builder.build_observation(
            portfolio_state=portfolio_state,
            current_data=current_data,
            initial_balance=self.config.initial_balance,
            external_predictions=ext_pred
        )

    def _start_polling_fallback(self) -> None:
        """Démarrer un thread de polling yfinance (interval 1m) comme fallback live.
        Pousse périodiquement la dernière bougie dans la data_queue.
        """
        def _poll_loop(symbol: str, q: queue.Queue):
            # Première poussée immédiate pour éviter timeout
            while True:
                try:
                    # Récupère la dernière ligne 1m (ou 5m si 1m indisponible)
                    for interval in ("1m", "5m"):
                        hist = yf.Ticker(symbol).history(period="1d", interval=interval)
                        if hist is not None and len(hist) > 0:
                            last = hist.iloc[-1]
                            break
                    else:
                        raise ValueError("Pas de données retournées par yfinance")

                    self.current_data['open'] = float(last.get('Open', last.get('open', self.current_data['open'])))
                    self.current_data['high'] = float(last.get('High', last.get('high', self.current_data['high'])))
                    self.current_data['low'] = float(last.get('Low', last.get('low', self.current_data['low'])))
                    self.current_data['close'] = float(last.get('Close', last.get('close', self.current_data['close'])))
                    self.current_data['volume'] = float(last.get('Volume', last.get('volume', self.current_data['volume'])))
                    q.put(self.current_data.copy())
                except Exception:
                    # En cas d'erreur réseau/parse, on réessaie
                    pass
                time.sleep(self.config.polling_interval_seconds)

        t = threading.Thread(target=_poll_loop, args=(self.symbol, self.data_queue))
        t.daemon = True
        t.start()
    
    def render(self, mode='human'):
        """Afficher l'état actuel de l'environnement."""
        if self.live:
            current_price = self.current_data["close"]
            step_max = self.config.max_steps_live
        else:
            current_price = self.data.iloc[self.current_step]["close"]
            step_max = len(self.data) - 1
        
        portfolio_state = self.portfolio.get_state(current_price)
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{step_max} (Mode: {'Live' if self.live else 'Historique'})")
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

# 6. FONCTIONS UTILITAIRES POUR CHARGER LES DONNÉES
def load_stock_data_from_csv(csv_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Charger les données boursières depuis un fichier CSV.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        symbol: Symbole spécifique à filtrer (optionnel)
    
    Returns:
        DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
        
    Format CSV attendu:
        Symbol,Date,Open,High,Low,Close,Volume
        AAPL,2020-01-02,71.55,72.6,71.29,72.54,135480400
        ...
    """
    try:
        print(f"Chargement des données depuis: {csv_path}")
        data = pd.read_csv(csv_path)
        
        required_columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le CSV: {missing_columns}")
        
        if symbol:
            symbol = symbol.upper()
            if symbol not in data['Symbol'].values:
                available_symbols = sorted(data['Symbol'].unique())
                print(f"⚠️ Symbole '{symbol}' non trouvé dans le CSV. Symboles disponibles: {', '.join(available_symbols)}")
            data = data[data['Symbol'] == symbol].copy()
            print(f"Filtrage sur le symbole: {symbol}")
        
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        result_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        if symbol:
            print(f"Données chargées pour {symbol}:")
        else:
            symbols = sorted(data['Symbol'].unique()) if 'Symbol' in data.columns else ['Multiple']
            print(f"Données chargées pour {len(symbols)} symbole(s): {', '.join(symbols[:5])}")
            if len(symbols) > 5:
                print(f"    ... et {len(symbols)-5} autres symboles")
        
        print(f"Période: du {data['Date'].min().strftime('%Y-%m-%d')} au {data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Nombre de lignes: {len(result_data):,}")
        print(f"Prix moyen: ${result_data['close'].mean():.2f}")
        
        return result_data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du CSV: {str(e)}")

def get_available_symbols(csv_path: str) -> List[str]:
    """
    Obtenir la liste des symboles disponibles dans le CSV.
    
    Args:
        csv_path: Chemin vers le fichier CSV
    
    Returns:
        Liste des symboles disponibles
    """
    try:
        data = pd.read_csv(csv_path)
        if 'Symbol' not in data.columns:
            raise ValueError("Colonne 'Symbol' non trouvée dans le CSV")
        
        symbols = sorted(data['Symbol'].unique())
        return symbols
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture des symboles: {str(e)}")

# 7. EXEMPLE D'UTILISATION
if __name__ == "__main__":
    csv_path = "../datatset/top10_stocks_2025_clean_international.csv"
    
    if not os.path.exists(csv_path):
        print(f"Fichier CSV non trouvé: {csv_path}")
        print("Assurez-vous que le chemin est correct et que le fichier existe.")
        exit(1)
    
    print("Démarrage de l'environnement de trading avec données CSV\n")
    
    try:
        available_symbols = get_available_symbols(csv_path)
        print(f"Symboles disponibles ({len(available_symbols)}): {', '.join(available_symbols)}\n")
    except Exception as e:
        print(f"Erreur lors de la lecture des symboles: {e}")
        exit(1)
    
    try:
        symbol = "AAPL"
        print(f"Chargement des données pour {symbol}...")
        data = load_stock_data_from_csv(csv_path, symbol=symbol)
        print(f"Données chargées avec succès!\n")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        exit(1)
    
    config = TradingConfig(
        initial_balance=10_000,
        transaction_fee=0.001,
        reward_type="profit",
        include_technical_indicators=True,
        lookback_window=60,
        max_steps_live=100
    )
    
    print(" Création de l'environnement historique...")
    env_hist = TradingEnv(data=data, config=config)
    print(f"✅ Environnement créé avec succès!\n")
    # Créer l'environnement
    print("🔧 Création de l'environnement de trading...")
    env = TradingEnv(data=data, config=config)
    print(f"Environnement créé avec succès!\n")
    
    # Test historique (simplifié)
    obs = env_hist.reset()
    print(f"Test historique démarré. Observation initiale: {obs.shape}")
    
    print("\n TEST EN MODE LIVE AVEC WEBSOCKET (AAPL)")
    # Informations sur l'environnement
    print("📊 INFORMATIONS SUR L'ENVIRONNEMENT")
    print("=" * 50)
    print(f"Symbole:                {symbol}")
    print(f"Balance initiale:       ${config.initial_balance:,}")
    print(f"Frais de transaction:   {config.transaction_fee*100:.1f}%")
    print(f"Type de récompense:     {config.reward_type}")
    print(f"Taille observation:     {env.observation_space.shape[0]}")
    print(f"Actions possibles:      {env.action_space.n} (Hold, Buy, Sell, Short)")
    print(f"Données disponibles:    {len(data)} jours")
    print(f"Période:                {data.index[0]} à {data.index[-1]}")
    print("=" * 50)
    print()
    
    # Test de l'environnement avec des actions aléatoires
    print("TEST AVEC ACTIONS ALÉATOIRES")
    print("=" * 50)
    
    env_live = TradingEnv(config=config, live=True, symbol="AAPL")
    
    obs = env_live.reset()
    
    total_reward = 0
    action_names = ['HOLD', 'BUY', 'SELL', 'SHORT']
    
    print("Simulation de quelques étapes live (attente de mises à jour prix)...")
    print("-" * 80)
    print(f"{'Step':<6} {'Action':<6} {'Prix':<8} {'Reward':<12} {'Net Worth':<12} {'Cash':<10} {'Shares':<8}")
    print("-" * 80)
    
    for step in range(20):
        action = env_live.action_space.sample()
        obs, reward, done, info = env_live.step(action)
        total_reward += reward
        
        current_price = env_live.current_data["close"]
        
        print(f"{step+1:<6} {action_names[action]:<6} ${current_price:<7.2f} "
              f"{reward:<11.6f} ${info['net_worth']:<11.2f} "
              f"${info['cash']:<9.2f} {info['shares']:<7.2f}")
        
        if done:
            print(f"\nEpisode live terminé à l'étape {step+1}")
            break
    
    print("-" * 80)
    print(f"Récompense totale: {total_reward:.6f}")
    
    env_live.render()
    
    print(f"\nTest live terminé!")
    print(f"💡 En mode live, les étapes avancent sur chaque mise à jour prix. Ajustez max_steps_live pour plus long.")