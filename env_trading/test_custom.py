"""
Script personnalisé pour tester l'environnement de trading
"""

from trading_env import TradingEnv, TradingConfig, load_stock_data_from_csv, get_available_symbols
import numpy as np

def test_symbol(symbol, steps=50):
    """Tester l'environnement avec un symbole spécifique"""
    
    # Chemin vers le CSV
    csv_path = "../datatset/top10_stocks_2025_clean_international.csv"
    
    print(f"Test avec le symbole: {symbol}")
    print("="*50)
    
    # Charger les données
    try:
        data = load_stock_data_from_csv(csv_path, symbol=symbol)
    except Exception as e:
        print(f"Erreur: {e}")
        return
    
    # Configuration
    config = TradingConfig(
        initial_balance=10_000,
        transaction_fee=0.001,
        reward_type="profit"
    )
    
    # Créer l'environnement
    env = TradingEnv(data=data, config=config)
    
    # Reset
    obs = env.reset()
    
    # Variables de suivi
    total_reward = 0
    actions_count = [0, 0, 0, 0]  # HOLD, BUY, SELL, SHORT
    action_names = ['HOLD', 'BUY', 'SELL', 'SHORT']
    
    print(f"Balance initiale: ${config.initial_balance:,}")
    print(f"Données disponibles: {len(data)} jours")
    print(f"Simulation de {steps} étapes...\n")
    
    # Simulation
    for step in range(steps):
        # Stratégie simple : acheter si prix bas, vendre si prix haut
        current_price = data.iloc[env.current_step]["close"]
        avg_price = data["close"].rolling(20).mean().iloc[env.current_step]
        
        if current_price < avg_price * 0.98 and env.portfolio.cash > 1000:
            action = 1  # BUY
        elif current_price > avg_price * 1.02 and env.portfolio.shares_held > 0:
            action = 2  # SELL
        else:
            action = 0  # HOLD
        
        # Exécuter l'action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        actions_count[action] += 1
        
        # Affichage périodique
        if step % 10 == 0 or step == steps-1:
            print(f"Step {step+1:3}: {action_names[action]:<5} | "
                  f"Prix: ${current_price:7.2f} | "
                  f"Net Worth: ${info['net_worth']:10.2f} | "
                  f"ROI: {((info['net_worth'] - config.initial_balance) / config.initial_balance * 100):6.2f}%")
        
        if done:
            print(f"\nEpisode terminé à l'étape {step+1}")
            break
    
    # Résultats finaux
    final_net_worth = env.portfolio.get_net_worth(data.iloc[env.current_step-1]["close"])
    roi = (final_net_worth - config.initial_balance) / config.initial_balance * 100
    
    print("\n" + "="*60)
    print("RÉSULTATS FINAUX")
    print("="*60)
    print(f"Net Worth final:     ${final_net_worth:,.2f}")
    print(f"ROI:                 {roi:+.2f}%")
    print(f"Récompense totale:   {total_reward:.6f}")
    print(f"Transactions:        {len(env.get_portfolio_history())}")
    print(f"Actions exécutées:")
    for i, count in enumerate(actions_count):
        print(f"   {action_names[i]}: {count}")
    print("="*60)

if __name__ == "__main__":
    # Afficher les symboles disponibles
    csv_path = "../datatset/top10_stocks_2025_clean_international.csv"
    symbols = get_available_symbols(csv_path)
    print(f"Symboles disponibles: {', '.join(symbols)}\n")
    
    # Tester plusieurs symboles
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    for symbol in test_symbols:
        if symbol in symbols:
            test_symbol(symbol, steps=100)
            print("\n" + "=="*30 + "\n")
        else:
            print(f"Symbole {symbol} non disponible")