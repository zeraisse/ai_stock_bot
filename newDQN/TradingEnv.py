import os
import time
import queue
import threading
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:
    import yfinance as yf  # utilisé pour WebSocket/polling live
except Exception:  # pragma: no cover
    yf = None


class TradingEnv(gym.Env):
    """
    Environnement minimaliste de trading.

    Modes supportés:
    - Simple (compatibilité arrière): initialiser avec `price` (np.ndarray/list de prix) → obs=[balance, holding, price]
    - Historique CSV: initialiser avec `data` (DataFrame OHLCV) → obs idem (prix=close)
    - Live: initialiser avec live=True et symbol (ex: "AAPL") → obs idem, flux via WebSocket/polling yfinance

    Actions: 0=HOLD, 1=BUY, 2=SELL
    """

    def __init__(self, price=None, data=None, *, initial_balance=1000.0, transaction_fee=0.001,
                 live=False, symbol=None, max_steps_live=100, live_initial_timeout_seconds=30,
                 live_step_timeout_seconds=65, polling_interval_seconds=2):
        super().__init__()

        # Configuration
        self.initial_balance = float(initial_balance)
        self.transaction_fee = float(transaction_fee)
        self.max_steps_live = int(max_steps_live)
        self.live_initial_timeout_seconds = int(live_initial_timeout_seconds)
        self.live_step_timeout_seconds = int(live_step_timeout_seconds)
        self.polling_interval_seconds = int(polling_interval_seconds)

        # Données (simple/historique)
        self.live = bool(live)
        self.symbol = symbol.upper() if symbol else None
        self.data = None
        self.price = None

        if self.live:
            if self.symbol is None:
                raise ValueError("Symbole requis en mode live (ex: 'AAPL')")
            self.current_tick = None
            self.data_queue = queue.Queue()
            self._ws = None
            self._ws_thread = None
        else:
            if data is not None:
                if not isinstance(data, pd.DataFrame) or 'close' not in data.columns:
                    raise ValueError("'data' doit être un DataFrame contenant la colonne 'close'")
                self.data = data.reset_index(drop=True)
                self.price = self.data['close'].to_numpy(dtype=float)
            elif price is not None:
                self.price = np.asarray(price, dtype=float)
            else:
                raise ValueError("Fournir 'price' (array) ou 'data' (DataFrame) quand live=False")

        # Etats
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0.0

        # Espaces
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0.0

        if self.live:
            self._start_live_stream()
            price = self._await_first_live_price()
            self.current_tick = float(price)
            obs = np.array([self.balance, self.holding, self.current_tick], dtype=np.float32)
            info = {}
            return obs, info

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        price = self._current_price()
        return np.array([self.balance, self.holding, price], dtype=np.float32)

    def _current_price(self):
        if self.live:
            return float(self.current_tick if self.current_tick is not None else 0.0)
        return float(self.price[self.current_step])

    def step(self, action):
        current_price = self._current_price()
        done = False

        old_value = self.balance + self.holding * current_price

        # Exécution naïve avec frais
        if action == 1:  # BUY
            cost = current_price * (1 + self.transaction_fee)
            if self.balance >= cost:
                self.holding += 1.0
                self.balance -= cost
        elif action == 2:  # SELL
            if self.holding > 0.0:
                self.holding -= 1.0
                self.balance += current_price * (1 - self.transaction_fee)

        # Avancer
        if self.live:
            # Attendre le prochain tick
            try:
                tick = self.data_queue.get(timeout=self.live_step_timeout_seconds)
                self.current_tick = float(tick)
            except queue.Empty:
                # Utiliser la dernière valeur si aucun tick
                pass
            self.current_step += 1
            if self.current_step >= self.max_steps_live:
                done = True
        else:
            self.current_step += 1
            if self.current_step >= len(self.price) - 1:
                done = True

        next_price = self._current_price()
        new_value = self.balance + self.holding * next_price
        reward = float(new_value - old_value)

        obs = np.array([self.balance, self.holding, next_price], dtype=np.float32)
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    # ==== LIVE (WebSocket/polling via yfinance) ====
    def _start_live_stream(self):
        self._stop_live_stream()  # nettoyage si déjà lancé
        self.current_tick = None

        def message_handler(message):
            try:
                if isinstance(message, dict):
                    msg = message.get(self.symbol, message)
                    close_val = float(msg.get('price', msg.get('close', self.current_tick or 0.0)))
                    self.data_queue.put(close_val)
            except Exception:
                pass

        # Tente WebSocket yfinance si dispo; sinon polling direct
        if yf is not None:
            try:
                self._ws = yf.WebSocket()
                self._ws.subscribe([self.symbol])
                self._ws_thread = threading.Thread(target=self._ws.listen, args=(message_handler,))
                self._ws_thread.daemon = True
                self._ws_thread.start()
            except Exception:
                self._start_polling_fallback()
        else:
            self._start_polling_fallback()

        # Lancer un moniteur de fallback si silence
        def _silence_monitor():
            last_ts = time.time()
            while True:
                try:
                    time.sleep(1)
                    # Si aucun nouveau tick n'arrive pendant live_initial_timeout_seconds, démarrer polling
                    if (self.current_tick is None) and (time.time() - last_ts > self.live_initial_timeout_seconds):
                        self._start_polling_fallback()
                        break
                except Exception:
                    break

        t = threading.Thread(target=_silence_monitor)
        t.daemon = True
        t.start()

    def _stop_live_stream(self):  # pragma: no cover
        try:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
            self._ws = None
        except Exception:
            pass

    def _start_polling_fallback(self):
        def _poll_loop(symbol, q):
            while True:
                try:
                    if yf is None:
                        raise RuntimeError("yfinance non disponible pour le polling")
                    # Essaye 1m puis 5m
                    last = None
                    for interval in ("1m", "5m"):
                        hist = yf.Ticker(symbol).history(period="1d", interval=interval)
                        if hist is not None and len(hist) > 0:
                            last = hist.iloc[-1]
                            break
                    if last is not None:
                        close_val = float(last.get('Close', last.get('close', self.current_tick or 0.0)))
                        q.put(close_val)
                except Exception:
                    pass
                time.sleep(self.polling_interval_seconds)

        t = threading.Thread(target=_poll_loop, args=(self.symbol, self.data_queue))
        t.daemon = True
        t.start()

    def _await_first_live_price(self):
        try:
            price = self.data_queue.get(timeout=self.live_initial_timeout_seconds)
            return float(price)
        except queue.Empty:
            # démarrer fallback si pas déjà en cours
            self._start_polling_fallback()
            price = self.data_queue.get(timeout=self.live_initial_timeout_seconds)
            return float(price)


# ==== Utilitaires CSV (repris et simplifiés) ====
def load_stock_data_from_csv(csv_path, symbol=None):
    import pandas as _pd
    data = _pd.read_csv(csv_path)
    required = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    if symbol:
        data = data[data['Symbol'] == symbol.upper()].copy()
        if data.empty:
            raise ValueError(f"Symbole {symbol} introuvable dans le CSV")
    data['Date'] = _pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return data[['open', 'high', 'low', 'close', 'volume']].copy()


def get_available_symbols(csv_path):
    import pandas as _pd
    d = _pd.read_csv(csv_path)
    if 'Symbol' not in d.columns:
        raise ValueError("Colonne 'Symbol' absente du CSV")
    return sorted(d['Symbol'].unique())


if __name__ == "__main__":
    # Test historique + live (simple) pour le nouvel environnement
    csv_path = "../dataset/top10_stocks_2025.csv"
    if not os.path.exists(csv_path):
        print(f"Fichier CSV non trouvé: {csv_path}")
        raise SystemExit(1)

    print("Démarrage des tests de TradingEnv (nouvelle version)\n")

    # Liste des symboles
    try:
        symbols = get_available_symbols(csv_path)
        print(f"Symboles disponibles ({len(symbols)}): {', '.join(symbols[:9])}")
    except Exception as e:
        print(f"Erreur lecture symboles: {e}")
        raise SystemExit(1)

    # Historique pour AAPL
    symbol = "AAPL" if "AAPL" in symbols else symbols[0]
    print(f"\nChargement des données historiques pour {symbol}...")
    data = load_stock_data_from_csv(csv_path, symbol)
    print(f"Période: {len(data)} lignes, prix moyen close: ${data['close'].mean():.2f}")

    env_hist = TradingEnv(price=data['close'].values, initial_balance=10_000, transaction_fee=0.001)
    obs, _ = env_hist.reset()
    print(f"Observation initiale (historique): {obs.shape} -> {obs}")

    total_reward = 0.0
    action_names = ['HOLD', 'BUY', 'SELL']
    print("\nSimulation historique (20 étapes):")
    print("-" * 80)
    print(f"{'Step':<6} {'Action':<6} {'Prix':<8} {'Reward':<12} {'Cash':<10} {'Shares':<8}")
    print("-" * 80)
    for step in range(20):
        action = env_hist.action_space.sample()
        obs, reward, terminated, truncated, _ = env_hist.step(action)
        done = terminated or truncated
        total_reward += reward
        price = obs[2]
        print(f"{step+1:<6} {action_names[action]:<6} ${price:<7.2f} {reward:<11.6f} ${env_hist.balance:<9.2f} {env_hist.holding:<7.2f}")
        if done:
            break
    print("-" * 80)
    print(f"Récompense totale (historique): {total_reward:.6f}")

    # Live (si yfinance disponible)
    if yf is not None:
        try:
            print("\nTEST EN MODE LIVE (quelques étapes)\n" + "=" * 50)
            env_live = TradingEnv(live=True, symbol=symbol, initial_balance=10_000, max_steps_live=20)
            obs, _ = env_live.reset()
            total_reward = 0.0
            print("Attente/marche en live...")
            print("-" * 80)
            for step in range(20):
                action = env_live.action_space.sample()
                obs, reward, terminated, truncated, _ = env_live.step(action)
                total_reward += reward
                price = obs[2]
                print(f"{step+1:<6} {action_names[action]:<6} ${price:<7.2f} {reward:<11.6f} ${env_live.balance:<9.2f} {env_live.holding:<7.2f}")
                if terminated or truncated:
                    break
            print("-" * 80)
            print(f"Récompense totale (live): {total_reward:.6f}")
        except Exception as e:
            print(f"Live non disponible: {e}")
    else:
        print("yfinance non disponible - test live ignoré")

