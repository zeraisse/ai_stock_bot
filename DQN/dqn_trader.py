"""
Script principal pour l'entraînement DQN Trading
==============================================

Ce script intègre l'agent DQN avec l'environnement de trading pour un entraînement
visible en temps réel avec des métriques de performance.

Usage:
    python dqn_trader.py [--symbol SYMBOL] [--episodes EPISODES] [--visual]
"""

import sys
import os
import argparse
import time
import numpy as np
from datetime import datetime
import threading

# Ajouter le dossier parent au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env_trading'))

from dqn_agent import DQNAgent, TradingVisualizer, get_action_name
from env_trading.trading_env import TradingEnv, TradingConfig, load_stock_data_from_csv, get_available_symbols


class DQNTrader:
    """
    Trader DQN principal qui coordonne l'entraînement
    """
    
    def __init__(self, 
                 symbol="AAPL",
                 episodes=1000,
                 save_interval=100,
                 visual=True,
                 model_dir="models"):
        
        self.symbol = symbol
        self.episodes = episodes
        self.save_interval = save_interval
        self.visual = visual
        self.model_dir = model_dir
        
        # Créer le dossier de modèles
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialiser l'environnement
        self.setup_environment()
        
        # Initialiser l'agent DQN
        self.setup_agent()
        
        # Visualiseur (optionnel)
        self.visualizer = None
        if visual:
            self.setup_visualizer()
        
        # Métriques d'entraînement
        self.training_stats = {
            'start_time': None,
            'best_reward': float('-inf'),
            'best_net_worth': 0,
            'total_trades': 0,
            'profitable_episodes': 0
        }
    
    def setup_environment(self):
        """Configurer l'environnement de trading"""
        print(f"Configuration de l'environnement pour {self.symbol}...")
        
        # Chemin vers les données
        csv_path = "../datatset/top10_stocks_2025_clean_international.csv"
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
        
        # Charger les données
        try:
            self.data = load_stock_data_from_csv(csv_path, symbol=self.symbol)
            print(f"Données chargées: {len(self.data)} jours")
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            raise
        
        # Configuration de l'environnement
        self.config = TradingConfig(
            initial_balance=10_000,
            transaction_fee=0.001,
            reward_type="profit",
            include_technical_indicators=True,
            lookback_window=60
        )
        
        # Créer l'environnement
        self.env = TradingEnv(data=self.data, config=self.config)
        
        print(f"Environnement configuré:")
        print(f"   - Balance initiale: ${self.config.initial_balance:,}")
        print(f"   - Frais: {self.config.transaction_fee*100:.1f}%")
        print(f"   - Observations: {self.env.observation_space.shape[0]} dimensions")
        print(f"   - Actions: {self.env.action_space.n} possibles")
    
    def setup_agent(self):
        """Configurer l'agent DQN"""
        print("Configuration de l'agent DQN...")
        
        self.agent = DQNAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            lr=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_size=50000,
            batch_size=32,
            target_update=100
        )
        
        # Essayer de charger un modèle existant
        model_path = os.path.join(self.model_dir, f"dqn_{self.symbol.lower()}_latest.pth")
        if os.path.exists(model_path):
            if self.agent.load_model(model_path):
                print(f"Modèle existant chargé, reprise de l'entraînement...")
        
        print(f"Agent DQN configuré:")
        print(f"   - Learning rate: {self.agent.lr}")
        print(f"   - Gamma: {self.agent.gamma}")
        print(f"   - Epsilon: {self.agent.epsilon:.4f}")
        print(f"   - Memory: {len(self.agent.memory)}/{self.agent.memory.buffer.maxlen}")
    
    def setup_visualizer(self):
        """Configurer le visualiseur"""
        print("Configuration de la visualisation...")
        self.visualizer = TradingVisualizer(self.agent, update_interval=1000)
        
        # Démarrer l'animation dans un thread séparé
        def start_viz():
            self.visualizer.start_animation()
        
        viz_thread = threading.Thread(target=start_viz, daemon=True)
        viz_thread.start()
        
        print("Visualisation démarrée")
    
    def train_episode(self, episode):
        """Entraîner un épisode"""
        state = self.env.reset()
        total_reward = 0
        episode_trades = 0
        step = 0
        
        while True:
            # Choisir une action
            action, q_values = self.agent.act(state, training=True)
            
            # Exécuter l'action
            next_state, reward, done, info = self.env.step(action)
            
            # Stocker l'expérience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Entraîner le réseau
            loss = self.agent.replay()
            
            # Mise à jour
            state = next_state
            total_reward += reward
            
            # Compter les trades
            if info.get('trade_result', {}).get('success', False):
                if info['trade_result']['type'] in ['BUY', 'SELL']:
                    episode_trades += 1
            
            step += 1
            
            if done:
                break
        
        # Métriques finales de l'épisode
        final_net_worth = info['net_worth']
        roi = ((final_net_worth - self.config.initial_balance) / self.config.initial_balance) * 100
        
        # Mettre à jour l'historique de l'agent
        self.agent.update_history(
            episode_reward=total_reward,
            net_worth=final_net_worth,
            action=action,
            q_values=q_values,
            loss=loss
        )
        
        # Mettre à jour les statistiques
        self.update_training_stats(total_reward, final_net_worth, episode_trades, roi)
        
        return {
            'episode': episode,
            'reward': total_reward,
            'net_worth': final_net_worth,
            'roi': roi,
            'trades': episode_trades,
            'steps': step,
            'epsilon': self.agent.epsilon,
            'loss': loss
        }
    
    def update_training_stats(self, reward, net_worth, trades, roi):
        """Mettre à jour les statistiques d'entraînement"""
        if reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = reward
        
        if net_worth > self.training_stats['best_net_worth']:
            self.training_stats['best_net_worth'] = net_worth
        
        self.training_stats['total_trades'] += trades
        
        if roi > 0:
            self.training_stats['profitable_episodes'] += 1
    
    def print_episode_stats(self, stats):
        """Afficher les statistiques de l'épisode"""
        episode = stats['episode']
        
        # Affichage conditionnel (tous les 10 épisodes ou épisodes importants)
        if episode % 10 == 0 or episode < 10 or stats['roi'] > 5:
            print(f"Episode {episode:4d} | "
                  f"Reward: {stats['reward']:8.4f} | "
                  f"Net Worth: ${stats['net_worth']:8.2f} | "
                  f"ROI: {stats['roi']:6.2f}% | "
                  f"Trades: {stats['trades']:2d} | "
                  f"Steps: {stats['steps']:3d} | "
                  f"ε: {stats['epsilon']:.3f}")
        
        # Affichage spécial pour les performances exceptionnelles
        if stats['roi'] > 10:
            print(f"EXCELLENT EPISODE {episode}! ROI: {stats['roi']:.2f}%")
        elif stats['roi'] > 5:
            print(f"Bon épisode {episode}! ROI: {stats['roi']:.2f}%")
    
    def save_progress(self, episode):
        """Sauvegarder le progrès"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder le modèle
        model_path = os.path.join(self.model_dir, f"dqn_{self.symbol.lower()}_latest.pth")
        checkpoint_path = os.path.join(self.model_dir, f"dqn_{self.symbol.lower()}_ep{episode}_{timestamp}.pth")
        
        self.agent.save_model(model_path)
        
        # Checkpoint périodique
        if episode % (self.save_interval * 5) == 0:
            self.agent.save_model(checkpoint_path)
        
        # Sauvegarder les graphiques
        if self.visualizer:
            plot_path = os.path.join(self.model_dir, f"training_plots_{self.symbol.lower()}_{timestamp}.png")
            self.visualizer.save_plots(plot_path)
    
    def print_training_summary(self):
        """Afficher le résumé de l'entraînement"""
        duration = time.time() - self.training_stats['start_time']
        profit_rate = (self.training_stats['profitable_episodes'] / self.episodes) * 100
        
        print("\n" + "="*80)
        print("RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("="*80)
        print(f"Symbole:                 {self.symbol}")
        print(f"Durée:                   {duration/60:.1f} minutes")
        print(f"Episodes:                {self.episodes}")
        print(f"Meilleure récompense:    {self.training_stats['best_reward']:.4f}")
        print(f"Meilleur Net Worth:      ${self.training_stats['best_net_worth']:.2f}")
        print(f"Episodes profitables:    {self.training_stats['profitable_episodes']}/{self.episodes} ({profit_rate:.1f}%)")
        print(f"Total trades:            {self.training_stats['total_trades']}")
        print(f"Epsilon final:           {self.agent.epsilon:.4f}")
        print(f"Mémoire utilisée:        {len(self.agent.memory)}/{self.agent.memory.buffer.maxlen}")
        print("="*80)
    
    def train(self):
        """Lancer l'entraînement complet"""
        print(f"\nDÉMARRAGE DE L'ENTRAÎNEMENT DQN")
        print(f"Symbole: {self.symbol}")
        print(f"Episodes: {self.episodes}")
        print(f"Visualisation: {'Activée' if self.visual else 'Désactivée'}")
        print("="*60)
        
        self.training_stats['start_time'] = time.time()
        
        try:
            for episode in range(1, self.episodes + 1):
                # Entraîner un épisode
                stats = self.train_episode(episode)
                
                # Afficher les statistiques
                self.print_episode_stats(stats)
                
                # Sauvegarder périodiquement
                if episode % self.save_interval == 0:
                    self.save_progress(episode)
                    print(f"Modèle sauvegardé (épisode {episode})")
                
                # Pause pour la visualisation
                if self.visual and episode % 10 == 0:
                    time.sleep(0.1)  # Petite pause pour la visualisation
            
            # Sauvegarde finale
            self.save_progress(self.episodes)
            
            # Résumé final
            self.print_training_summary()
            
        except KeyboardInterrupt:
            print(f"\nEntraînement interrompu à l'épisode {episode}")
            self.save_progress(episode)
            self.print_training_summary()
        
        finally:
            if self.visualizer:
                print("Visualisation en cours... Fermez la fenêtre pour terminer.")
                try:
                    import matplotlib.pyplot as plt
                    plt.show()  # Garder la fenêtre ouverte
                except:
                    pass


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Entraînement DQN pour Trading")
    parser.add_argument('--symbol', type=str, default='AAPL', 
                       help='Symbole boursier à trader (défaut: AAPL)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Nombre d\'épisodes d\'entraînement (défaut: 500)')
    parser.add_argument('--no-visual', action='store_true',
                       help='Désactiver la visualisation en temps réel')
    
    args = parser.parse_args()
    
    # Vérifier que le symbole est disponible
    try:
        csv_path = "../datatset/top10_stocks_2025_clean_international.csv"
        available_symbols = get_available_symbols(csv_path)
        
        if args.symbol.upper() not in available_symbols:
            print(f"Symbole '{args.symbol}' non disponible.")
            print(f"Symboles disponibles: {', '.join(available_symbols)}")
            return
    except Exception as e:
        print(f"Erreur lors de la vérification des symboles: {e}")
        return
    
    # Créer et lancer le trader
    trader = DQNTrader(
        symbol=args.symbol.upper(),
        episodes=args.episodes,
        visual=not args.no_visual
    )
    
    trader.train()


if __name__ == "__main__":
    main()