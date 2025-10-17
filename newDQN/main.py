import os
import argparse
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import TradingEnv as te
import DQNAgent as dqn

try:
    # PyTorch PPO agent (preferred)
    from PPOAgentPT import train_ppo_with_env
except Exception:
    train_ppo_with_env = None

try:
    # NEAT training entry (optional)
    from neat_train import run_neat
except Exception:
    run_neat = None


def resolve_csv_path(path: str) -> str:
    # Absolute path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = []
    # Relative to current working directory
    candidates.append(os.path.abspath(path))
    # Relative to project root (parent of this file's dir)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    norm_rel = path.lstrip('./\\')
    candidates.append(os.path.join(project_root, norm_rel))
    # Common default within project
    candidates.append(os.path.join(project_root, 'dataset', 'top10_stocks_2025.csv'))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"CSV not found. Tried: {candidates}. Provide --csv with a valid path.")


def load_prices(csv_path: str):
    csv_path = resolve_csv_path(csv_path)
    data = pd.read_csv(csv_path)
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices).flatten()
    return prices


def run_dqn(env, prices, episodes: int = 10):
    agent = dqn.DQNAgent(state_size=3, action_size=3)
    for episode in range(episodes):
        state = env.reset()
        state = np.array(state[0])
        state = np.reshape(state, [1, 3])
        total_reward = 0
        for _ in range(len(prices) - 1):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            next_state = np.reshape(next_state, [1, 3])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")
                break
        if len(agent.memory) > 32:
            agent.replay(32)
    agent.save("models/dqn_trading_model.h5")


def run_ppo(env, epochs: int = 200, steps_per_epoch: int = 2000):
    if train_ppo_with_env is None:
        raise RuntimeError("PyTorch PPO not available. Ensure PPOAgentPT.py is present and importable.")
    agent, history = train_ppo_with_env(env, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # Plot PPO training history
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(history, label='Mean episode return per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Return')
        plt.title('PPO Training')
        plt.legend()
        os.makedirs('./models', exist_ok=True)
        plt.tight_layout()
        plt.savefig('./models/ppo_training.png')
        plt.close()
    except Exception:
        pass
    return agent, history


def run_neat_cli(generations: int = 20):
    if run_neat is None:
        raise RuntimeError("neat-python not available. Ensure neat_train.py and neat-python are installed.")
    config_path = os.path.join(os.path.dirname(__file__), 'neat-config.ini')
    winner, stats, fitness_history = run_neat(config_path, generations=generations)
    # Plot NEAT best fitness per generation
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(fitness_history, label='Best fitness per generation')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('NEAT Training')
        plt.legend()
        os.makedirs('./models', exist_ok=True)
        plt.tight_layout()
        plt.savefig('./models/neat_training.png')
        plt.close()
    except Exception:
        pass
    return winner, stats, fitness_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['dqn', 'ppo', 'neat'], default='dqn')
    parser.add_argument('--csv', default='../dataset/top10_stocks_2025.csv')
    # PPO controls
    parser.add_argument('--ppo-epochs', type=int, default=200)
    parser.add_argument('--ppo-steps', type=int, default=2000)
    # NEAT controls
    parser.add_argument('--neat-generations', type=int, default=50)
    args = parser.parse_args()

    os.makedirs('./models', exist_ok=True)

    prices = load_prices(args.csv)
    env = te.TradingEnv(prices)

    if args.algo == 'dqn':
        run_dqn(env, prices)
    elif args.algo == 'ppo':
        run_ppo(env, epochs=args.ppo_epochs, steps_per_epoch=args.ppo_steps)
    else:
        run_neat_cli(generations=args.neat_generations)


if __name__ == '__main__':
    main()
