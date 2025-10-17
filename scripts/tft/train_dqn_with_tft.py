"""
Script d'intégration DQN + TFT pour le trading algorithmique.

Ce script :
1. Charge ou entraîne un modèle TFT
2. Génère des prédictions TFT
3. Intègre les prédictions dans l'environnement de trading
4. Entraîne un agent DQN avec ces prédictions
5. Compare les performances DQN vs DQN+TFT
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Ajout des chemins
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.tft import TFTPredictor
from env_trading.trading_env import TradingEnv, TradingConfig, load_stock_data_from_csv
from DQN.dqn_agent import DQNAgent


class TFTDQNIntegration:
    """
    Classe pour gérer l'intégration TFT + DQN.
    """
    
    def __init__(self, 
                 symbol='AAPL',
                 csv_path='dataset/top10_stocks_2025.csv',
                 tft_model_path=None,
                 dqn_model_path=None):
        """
        Initialise l'intégration TFT + DQN.
        
        Args:
            symbol: Symbole boursier
            csv_path: Chemin vers le CSV
            tft_model_path: Chemin vers modèle TFT (None = entraîner)
            dqn_model_path: Chemin vers modèle DQN (None = entraîner)
        """
        self.symbol = symbol
        self.csv_path = csv_path
        self.tft_model_path = tft_model_path
        self.dqn_model_path = dqn_model_path
        
        # Composants
        self.data = None
        self.tft_predictor = None
        self.tft_predictions = None
        self.env_without_tft = None
        self.env_with_tft = None
        self.dqn_agent_baseline = None
        self.dqn_agent_tft = None
        
        print("="*80)
        print(f"INTÉGRATION TFT + DQN POUR {symbol}")
        print("="*80)
    
    def load_data(self):
        """
        Charge les données boursières.
        """
        print(f"\n📊 ÉTAPE 1 : Chargement des données")
        print("-"*80)
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Fichier non trouvé : {self.csv_path}")
        
        self.data = load_stock_data_from_csv(self.csv_path, symbol=self.symbol)
        
        print(f"✓ Données chargées : {len(self.data)} jours")
        print(f"  Période : {self.data.index[0]} à {self.data.index[-1]}")
        print(f"  Prix moyen : ${self.data['close'].mean():.2f}")
        
        return self.data
    
    def prepare_or_load_tft(self, 
                           epochs=50, 
                           batch_size=64,
                           force_retrain=False):
        """
        Prépare ou charge le modèle TFT.
        
        Args:
            epochs: Nombre d'époques si entraînement
            batch_size: Taille des batchs
            force_retrain: Force le réentraînement même si modèle existe
        """
        print(f"\n🤖 ÉTAPE 2 : Préparation du modèle TFT")
        print("-"*80)
        
        # Création du prédicteur
        self.tft_predictor = TFTPredictor(
            max_encoder_length=60,
            max_prediction_length=1,
            hidden_size=64,
            lstm_layers=2,
            attention_head_size=4,
            dropout=0.1,
            learning_rate=0.001
        )
        
        # Vérification si modèle existe
        model_exists = self.tft_model_path and os.path.exists(self.tft_model_path)
        
        if model_exists and not force_retrain:
            print(f"📦 Chargement du modèle existant : {self.tft_model_path}")
            
            # Préparation des données (nécessaire pour load)
            X, y = self.tft_predictor.prepare_data(self.data)
            
            # Chargement du modèle
            self.tft_predictor.load(self.tft_model_path)
            print("✓ Modèle TFT chargé avec succès")
            
        else:
            print(f"🎓 Entraînement d'un nouveau modèle TFT...")
            
            # Préparation des données
            X, y = self.tft_predictor.prepare_data(self.data)
            
            # Entraînement
            history = self.tft_predictor.train(
                X_train=X, y_train=y,
                X_val=X, y_val=y,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Sauvegarde
            if self.tft_model_path is None:
                self.tft_model_path = f"saved_models/tft_{self.symbol.lower()}.ckpt"
            
            os.makedirs(os.path.dirname(self.tft_model_path), exist_ok=True)
            self.tft_predictor.save(self.tft_model_path)
            
            print(f"✓ Modèle TFT entraîné et sauvegardé : {self.tft_model_path}")
        
        return self.tft_predictor
    
    def generate_tft_predictions(self):
        """
        Génère les prédictions TFT.
        """
        print(f"\n🔮 ÉTAPE 3 : Génération des prédictions TFT")
        print("-"*80)
        
        if self.tft_predictor is None:
            raise ValueError("Le prédicteur TFT n'est pas initialisé")
        
        # Génération
        self.tft_predictions = self.tft_predictor.predict(np.array([0]))  # Placeholder
        
        print(f"✓ Prédictions générées : {self.tft_predictions.shape}")
        print(f"  Format : (n_samples, 3) = [prix_ratio, tendance, volatilité]")
        
        # Statistiques
        print(f"\n  Statistiques des prédictions :")
        print(f"  - Prix ratio   : [{self.tft_predictions[:, 0].min():.4f}, {self.tft_predictions[:, 0].max():.4f}]")
        print(f"  - Tendance     : [{self.tft_predictions[:, 1].min():.4f}, {self.tft_predictions[:, 1].max():.4f}]")
        print(f"  - Volatilité   : [{self.tft_predictions[:, 2].min():.4f}, {self.tft_predictions[:, 2].max():.4f}]")
        
        # Sauvegarde optionnelle
        pred_path = f"predictions/tft_predictions_{self.symbol.lower()}.npy"
        os.makedirs("predictions", exist_ok=True)
        np.save(pred_path, self.tft_predictions)
        print(f"✓ Prédictions sauvegardées : {pred_path}")
        
        return self.tft_predictions
    
    def create_environments(self):
        """
        Crée les environnements de trading (avec et sans TFT).
        """
        print(f"\n🏗️ ÉTAPE 4 : Création des environnements de trading")
        print("-"*80)
        
        # Configuration commune
        config = TradingConfig(
            initial_balance=10_000,
            transaction_fee=0.001,  # 0.1%
            reward_type="profit",
            include_technical_indicators=True
        )
        
        # Environnement SANS TFT (baseline)
        print("  Création environnement baseline (sans TFT)...")
        self.env_without_tft = TradingEnv(data=self.data, config=config)
        obs_size_baseline = self.env_without_tft.observation_space.shape[0]
        print(f"  ✓ Environnement baseline créé - Observation size: {obs_size_baseline}")
        
        # Environnement AVEC TFT
        print("  Création environnement avec prédictions TFT...")
        self.env_with_tft = TradingEnv(
            data=self.data, 
            config=config,
            external_predictions=self.tft_predictions
        )
        obs_size_tft = self.env_with_tft.observation_space.shape[0]
        print(f"  ✓ Environnement TFT créé - Observation size: {obs_size_tft}")
        
        print(f"\n  Différence d'observation : +{obs_size_tft - obs_size_baseline} features (TFT)")
        
        return self.env_without_tft, self.env_with_tft
    
    def train_dqn_agents(self, 
                        episodes=1000,
                        save_interval=100,
                        compare_baseline=True):
        """
        Entraîne les agents DQN (avec et sans TFT).
        
        Args:
            episodes: Nombre d'épisodes
            save_interval: Interval de sauvegarde
            compare_baseline: Si True, entraîne aussi l'agent baseline
        """
        print(f"\n🎮 ÉTAPE 5 : Entraînement des agents DQN")
        print("-"*80)
        
        results = {}
        
        # Agent avec TFT
        print(f"\n  🚀 Entraînement DQN + TFT...")
        self.dqn_agent_tft = DQNAgent(
            state_size=self.env_with_tft.observation_space.shape[0],
            action_size=self.env_with_tft.action_space.n,
            lr=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_size=50000,
            batch_size=32
        )
        
        results['dqn_tft'] = self._train_single_agent(
            agent=self.dqn_agent_tft,
            env=self.env_with_tft,
            episodes=episodes,
            name="DQN+TFT",
            save_path=f"saved_models/dqn_tft_{self.symbol.lower()}.pth",
            save_interval=save_interval
        )
        
        # Agent baseline (optionnel)
        if compare_baseline:
            print(f"\n  📊 Entraînement DQN Baseline (sans TFT)...")
            self.dqn_agent_baseline = DQNAgent(
                state_size=self.env_without_tft.observation_space.shape[0],
                action_size=self.env_without_tft.action_space.n,
                lr=0.001,
                gamma=0.95,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.995,
                memory_size=50000,
                batch_size=32
            )
            
            results['dqn_baseline'] = self._train_single_agent(
                agent=self.dqn_agent_baseline,
                env=self.env_without_tft,
                episodes=episodes,
                name="DQN Baseline",
                save_path=f"saved_models/dqn_baseline_{self.symbol.lower()}.pth",
                save_interval=save_interval
            )
        
        return results
    
    def _train_single_agent(self, agent, env, episodes, name, save_path, save_interval):
        """
        Entraîne un seul agent DQN.
        """
        rewards_history = []
        net_worth_history = []
        epsilon_history = []
        
        print(f"  Début entraînement {name} sur {episodes} épisodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Sélection action
                action = agent.act(state)
                
                # Step environnement
                next_state, reward, done, info = env.step(action)
                
                # Mémorisation
                agent.remember(state, action, reward, next_state, done)
                
                # Apprentissage
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                total_reward += reward
            
            # Historique
            rewards_history.append(total_reward)
            net_worth_history.append(info['net_worth'])
            epsilon_history.append(agent.epsilon)
            
            # Affichage périodique
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                avg_net_worth = np.mean(net_worth_history[-50:])
                print(f"    Episode {episode+1}/{episodes} | "
                      f"Reward: {avg_reward:.6f} | "
                      f"Net Worth: ${avg_net_worth:,.2f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
            
            # Sauvegarde périodique
            if (episode + 1) % save_interval == 0:
                agent.save(save_path)
        
        # Sauvegarde finale
        agent.save(save_path)
        print(f"  ✓ {name} entraîné - Modèle sauvegardé : {save_path}")
        
        return {
            'rewards': rewards_history,
            'net_worth': net_worth_history,
            'epsilon': epsilon_history
        }
    
    def compare_results(self, results):
        """
        Compare les résultats DQN vs DQN+TFT.
        """
        print(f"\n📈 ÉTAPE 6 : Comparaison des performances")
        print("="*80)
        
        if 'dqn_baseline' in results and 'dqn_tft' in results:
            baseline = results['dqn_baseline']
            tft = results['dqn_tft']
            
            # Métriques finales
            print(f"\n🏆 RÉSULTATS FINAUX (derniers 50 épisodes)")
            print("-"*80)
            
            baseline_avg_reward = np.mean(baseline['rewards'][-50:])
            tft_avg_reward = np.mean(tft['rewards'][-50:])
            
            baseline_avg_worth = np.mean(baseline['net_worth'][-50:])
            tft_avg_worth = np.mean(tft['net_worth'][-50:])
            
            print(f"DQN Baseline :")
            print(f"  - Récompense moyenne : {baseline_avg_reward:.6f}")
            print(f"  - Net Worth moyen    : ${baseline_avg_worth:,.2f}")
            
            print(f"\nDQN + TFT :")
            print(f"  - Récompense moyenne : {tft_avg_reward:.6f}")
            print(f"  - Net Worth moyen    : ${tft_avg_worth:,.2f}")
            
            # Amélioration
            reward_improvement = ((tft_avg_reward - baseline_avg_reward) / abs(baseline_avg_reward)) * 100
            worth_improvement = ((tft_avg_worth - baseline_avg_worth) / baseline_avg_worth) * 100
            
            print(f"\n📊 AMÉLIORATION TFT vs Baseline :")
            print(f"  - Récompense : {reward_improvement:+.2f}%")
            print(f"  - Net Worth  : {worth_improvement:+.2f}%")
            
            # Visualisation
            self._plot_comparison(baseline, tft)
        else:
            print("⚠ Pas assez de données pour comparaison")
    
    def _plot_comparison(self, baseline, tft):
        """
        Crée des graphiques de comparaison.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Comparaison DQN vs DQN+TFT - {self.symbol}', fontsize=16)
        
        # 1. Récompenses
        axes[0, 0].plot(baseline['rewards'], label='DQN Baseline', alpha=0.7)
        axes[0, 0].plot(tft['rewards'], label='DQN + TFT', alpha=0.7)
        axes[0, 0].set_title('Récompenses par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Net Worth
        axes[0, 1].plot(baseline['net_worth'], label='DQN Baseline', alpha=0.7)
        axes[0, 1].plot(tft['net_worth'], label='DQN + TFT', alpha=0.7)
        axes[0, 1].axhline(y=10000, color='r', linestyle='--', label='Balance initiale')
        axes[0, 1].set_title('Net Worth par épisode')
        axes[0, 1].set_xlabel('Épisode')
        axes[0, 1].set_ylabel('Net Worth ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Moyenne mobile récompenses
        window = 50
        baseline_ma = pd.Series(baseline['rewards']).rolling(window).mean()
        tft_ma = pd.Series(tft['rewards']).rolling(window).mean()
        axes[1, 0].plot(baseline_ma, label='DQN Baseline (MA50)', linewidth=2)
        axes[1, 0].plot(tft_ma, label='DQN + TFT (MA50)', linewidth=2)
        axes[1, 0].set_title('Récompenses - Moyenne Mobile (50 épisodes)')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Récompense moyenne')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribution finale Net Worth
        final_baseline = baseline['net_worth'][-100:]
        final_tft = tft['net_worth'][-100:]
        axes[1, 1].hist(final_baseline, bins=20, alpha=0.5, label='DQN Baseline')
        axes[1, 1].hist(final_tft, bins=20, alpha=0.5, label='DQN + TFT')
        axes[1, 1].axvline(np.mean(final_baseline), color='blue', linestyle='--', linewidth=2)
        axes[1, 1].axvline(np.mean(final_tft), color='orange', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Distribution Net Worth (100 derniers épisodes)')
        axes[1, 1].set_xlabel('Net Worth ($)')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarde
        plot_path = f"results/comparison_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphiques sauvegardés : {plot_path}")
        
        plt.show()
    
    def run_full_pipeline(self, 
                         tft_epochs=50,
                         dqn_episodes=1000,
                         force_retrain_tft=False,
                         compare_baseline=True):
        """
        Exécute le pipeline complet TFT + DQN.
        """
        print("\n" + "="*80)
        print("🚀 DÉMARRAGE DU PIPELINE COMPLET TFT + DQN")
        print("="*80)
        
        # 1. Charger données
        self.load_data()
        
        # 2. TFT
        self.prepare_or_load_tft(epochs=tft_epochs, force_retrain=force_retrain_tft)
        
        # 3. Prédictions
        self.generate_tft_predictions()
        
        # 4. Environnements
        self.create_environments()
        
        # 5. DQN
        results = self.train_dqn_agents(episodes=dqn_episodes, compare_baseline=compare_baseline)
        
        # 6. Comparaison
        self.compare_results(results)
        
        print("\n" + "="*80)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("="*80)
        
        return results


def main():
    """
    Point d'entrée principal.
    """
    parser = argparse.ArgumentParser(description='Intégration TFT + DQN pour trading')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbole boursier (défaut: AAPL)')
    parser.add_argument('--csv', type=str, 
                       default='dataset/top10_stocks_2025.csv',
                       help='Chemin vers CSV')
    parser.add_argument('--tft_epochs', type=int, default=50,
                       help='Époques TFT (défaut: 50)')
    parser.add_argument('--dqn_episodes', type=int, default=1000,
                       help='Épisodes DQN (défaut: 1000)')
    parser.add_argument('--force_retrain_tft', action='store_true',
                       help='Force réentraînement TFT')
    parser.add_argument('--no_baseline', action='store_true',
                       help='Ne pas entraîner baseline')
    
    args = parser.parse_args()
    
    # Création de l'intégration
    integration = TFTDQNIntegration(
        symbol=args.symbol.upper(),
        csv_path=args.csv
    )
    
    # Exécution du pipeline
    results = integration.run_full_pipeline(
        tft_epochs=args.tft_epochs,
        dqn_episodes=args.dqn_episodes,
        force_retrain_tft=args.force_retrain_tft,
        compare_baseline=not args.no_baseline
    )
    
    print("\n🎉 Intégration TFT + DQN terminée avec succès !")


if __name__ == "__main__":
    main()