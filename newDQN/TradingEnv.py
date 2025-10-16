import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, price):
        super().__init__()
        self.price = price
        self.current_step = 0
        self.balance = 1000
        self.holding = 0

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 1000
        self.holding = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array([self.balance, self.holding, self.price[self.current_step]], dtype=np.float32)

    def step(self, action):
        current_price = self.price[self.current_step]
        done = False

        old_value = self.balance + self.holding * current_price

        if action == 1:  # Acheter
            if self.balance >= current_price:
                self.holding += 1
                self.balance -= current_price
        elif action == 2:  # Vendre
            if self.holding > 0:
                self.holding -= 1
                self.balance += current_price

        # Avancer d’un pas
        self.current_step += 1
        if self.current_step >= len(self.price) - 1:
            done = True

        next_price = self.price[self.current_step]
        new_value = self.balance + self.holding * next_price
        reward = new_value - old_value

        obs = np.array([self.balance, self.holding, next_price], dtype=np.float32)

        terminated = done
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info
