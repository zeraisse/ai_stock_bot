import os
import argparse
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import TradingEnv as te
import DQNAgent as dqn
import glob
import matplotlib.pyplot as plt



try:
    from PPOAgentPT import train_ppo_with_env
except Exception:
    train_ppo_with_env = None

try:
    from neat_train import run_neat
except Exception:
    run_neat = None


def resolve_csv_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = []
    candidates.append(os.path.abspath(path))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    norm_rel = path.lstrip('./\\')
    candidates.append(os.path.join(project_root, norm_rel))
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

def save_models_dqn(episode, agent):
      if (episode + 1) % 100 == 0:
            backup_path = f"models/backup_models/dqn_trading_model_backup_ep{episode+1}.h5"
            agent.save(backup_path)
            print(f"Backup sauvegardé: {backup_path}")




def load_last_backup(agent, backup_dir="models/backup_models"):
    """
    Charge le dernier backup disponible pour l'agent DQN.
    
    Args:
        agent (DQNAgent): l'agent DQN
        backup_dir (str): dossier où sont stockés les backups
    
    Returns:
        episode_start (int): l'épisode à partir duquel continuer
    """
    backup_files = sorted(
        glob.glob(os.path.join(backup_dir, "dqn_trading_model_backup_ep*.h5")),
        key=os.path.getmtime
    )
    if backup_files:
        last_backup = backup_files[-1]
        agent.load(last_backup)
        episode_start = int(last_backup.split("_ep")[-1].split(".")[0])
        print(f"Dernier backup chargé : {last_backup}, reprise depuis épisode {episode_start + 1}")
        return episode_start + 1
    else:
        print("Aucun backup trouvé, début de l'entraînement depuis l'épisode 1")
        return 0



def run_dqn(env, prices, episodes: int = 100):
    os.makedirs("models/backup_models", exist_ok=True)
    agent = dqn.DQNAgent(state_size=3, action_size=3)
    start_episode = load_last_backup(agent)
    
    rewards = []

    for episode in range(start_episode, episodes):
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
                break

        rewards.append(total_reward)
        print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

        if len(agent.memory) > 128:
            agent.replay(128)
            

        save_models_dqn(episode, agent)

    agent.save("models/backup_models/dqn_trading_model_final.h5")
    display_rewards(rewards)


def display_rewards(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.title("Évolution du Total Reward pendant l'apprentissage")
    plt.xlabel("Épisodes")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("models/reward_plot.png")


def run_ppo(env):
    if train_ppo_with_env is None:
        raise RuntimeError("PyTorch PPO not available. Ensure PPOAgentPT.py is present and importable.")
    train_ppo_with_env(env, epochs=10, steps_per_epoch=2000)


def run_neat_cli():
    if run_neat is None:
        raise RuntimeError("neat-python not available. Ensure neat_train.py and neat-python are installed.")
    config_path = os.path.join(os.path.dirname(__file__), 'neat-config.ini')
    run_neat(config_path, generations=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['dqn', 'ppo', 'neat'], default='dqn')
    parser.add_argument('--csv', default='../dataset/top10_stocks_2025.csv')
    args = parser.parse_args()

    os.makedirs('./models', exist_ok=True)

    prices = load_prices(args.csv)
    env = te.TradingEnv(prices)

    if args.algo == 'dqn':
        run_dqn(env, prices)
    elif args.algo == 'ppo':
        run_ppo(env)
    else:
        run_neat_cli()


if __name__ == '__main__':
    main()
