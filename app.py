"""
Dashboard Streamlit pour visualiser les performances DQN-TFT vs DQN-LSTM.

Lancer avec: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import sys
import os
from pathlib import Path
import glob
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'newDQN'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from newDQN.DQNAgent import DQNAgent
from newDQN.TradingEnv import TradingEnv, load_stock_data_from_csv
from newDQN import DQNTFTModel, DQNLSTMModel

# Configuration page
st.set_page_config(
    page_title="DQN-TFT Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .success-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .danger-metric {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_training_history():
    """Je charge l'historique d'entrainement depuis les graphiques sauvegardes."""
    history = {}
    reward_files = glob.glob("newDQN/models/*.png")
    
    for file in reward_files:
        # Je parse le nom du fichier pour identifier le modele
        filename = Path(file).stem
        if 'reward_plot' in filename.lower():
            # Je charge les donnees si disponibles
            data_file = file.replace('.png', '_data.json')
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    history[filename] = json.load(f)
    
    return history


@st.cache_data
def load_stock_data(csv_path, symbol):
    """Je charge les donnees boursieres pour un symbole."""
    try:
        data = load_stock_data_from_csv(csv_path, symbol)
        data['date'] = pd.date_range(start='2024-01-01', periods=len(data), freq='D')
        return data
    except Exception as e:
        st.error(f"Erreur chargement données: {e}")
        return None


@st.cache_resource
def load_trained_agent(model_type, model_path):
    """Je charge un agent DQN entraine."""
    try:
        agent = DQNAgent(state_size=3, action_size=3)
        if os.path.exists(model_path):
            agent.load(model_path)
            return agent
        else:
            st.warning(f"Modèle {model_type} non trouvé: {model_path}")
            return None
    except Exception as e:
        st.error(f"Erreur chargement {model_type}: {e}")
        return None


def run_live_simulation(symbol, agent, duration_minutes=5):
    """Je lance une simulation en mode live avec WebSocket."""
    st.info(f"🔴 Mode LIVE activé pour {symbol} - Durée: {duration_minutes} min")
    
    # Je cree l'environnement live
    env = TradingEnv(
        live=True,
        symbol=symbol,
        initial_balance=10000,
        transaction_fee=0.001,
        max_steps_live=duration_minutes * 60  # 1 step par seconde
    )
    
    # Placeholders pour mise à jour temps réel
    price_chart = st.empty()
    metrics_container = st.empty()
    actions_log = st.empty()
    
    # Je stocke les données de simulation
    simulation_data = {
        'timestamps': [],
        'prices': [],
        'balance': [],
        'holdings': [],
        'actions': [],
        'net_worth': []
    }
    
    try:
        state = env.reset()
        state = np.array(state[0])
        state = np.reshape(state, [1, 3])
        
        step = 0
        done = False
        
        while not done and step < duration_minutes * 60:
            # Je recupere l'action de l'agent
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Je stocke les metriques
            simulation_data['timestamps'].append(datetime.now())
            simulation_data['prices'].append(next_state[2])
            simulation_data['balance'].append(env.balance)
            simulation_data['holdings'].append(env.holding)
            simulation_data['actions'].append(['HOLD', 'BUY', 'SELL'][action])
            simulation_data['net_worth'].append(env.balance + env.holding * next_state[2])
            
            # Je mets à jour l'affichage toutes les 5 secondes
            if step % 5 == 0:
                # Graphique prix
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=simulation_data['timestamps'],
                    y=simulation_data['prices'],
                    mode='lines',
                    name='Prix live',
                    line=dict(color='blue', width=2)
                ))
                
                # Je marque les actions BUY/SELL
                buy_indices = [i for i, a in enumerate(simulation_data['actions']) if a == 'BUY']
                sell_indices = [i for i, a in enumerate(simulation_data['actions']) if a == 'SELL']
                
                if buy_indices:
                    fig.add_trace(go.Scatter(
                        x=[simulation_data['timestamps'][i] for i in buy_indices],
                        y=[simulation_data['prices'][i] for i in buy_indices],
                        mode='markers',
                        name='BUY',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ))
                
                if sell_indices:
                    fig.add_trace(go.Scatter(
                        x=[simulation_data['timestamps'][i] for i in sell_indices],
                        y=[simulation_data['prices'][i] for i in sell_indices],
                        mode='markers',
                        name='SELL',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ))
                
                fig.update_layout(
                    title=f"Trading Live - {symbol}",
                    xaxis_title="Temps",
                    yaxis_title="Prix ($)",
                    height=400
                )
                price_chart.plotly_chart(fig, use_container_width=True)
                
                # Métriques
                with metrics_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Prix actuel", f"${next_state[2]:.2f}")
                    col2.metric("Cash", f"${env.balance:.2f}")
                    col3.metric("Actions détenues", f"{env.holding:.0f}")
                    col4.metric("Net Worth", f"${simulation_data['net_worth'][-1]:.2f}")
                
                # Log des actions
                recent_actions = list(zip(
                    simulation_data['timestamps'][-10:],
                    simulation_data['actions'][-10:],
                    simulation_data['prices'][-10:]
                ))
                actions_df = pd.DataFrame(recent_actions, columns=['Temps', 'Action', 'Prix'])
                actions_log.dataframe(actions_df, use_container_width=True)
            
            state = np.reshape(next_state, [1, 3])
            step += 1
        
        st.success("✅ Simulation live terminée")
        return pd.DataFrame(simulation_data)
    
    except Exception as e:
        st.error(f"❌ Erreur simulation live: {e}")
        return None


def run_backtest(data, agent, episodes=1):
    """Je lance un backtest sur donnees historiques."""
    env = TradingEnv(price=data['close'].values, initial_balance=10000)
    
    episode_data = {
        'steps': [],
        'dates': [],
        'balance': [],
        'holdings': [],
        'prices': [],
        'actions': [],
        'rewards': [],
        'net_worth': []
    }
    
    state = env.reset()
    state = np.array(state[0])
    state = np.reshape(state, [1, 3])
    
    step = 0
    done = False
    
    while not done and step < len(data) - 1:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_data['steps'].append(step)
        episode_data['dates'].append(data['date'].iloc[step] if 'date' in data.columns else step)
        episode_data['balance'].append(env.balance)
        episode_data['holdings'].append(env.holding)
        episode_data['prices'].append(next_state[2])
        episode_data['actions'].append(['HOLD', 'BUY', 'SELL'][action])
        episode_data['rewards'].append(reward)
        episode_data['net_worth'].append(env.balance + env.holding * next_state[2])
        
        state = np.reshape(next_state, [1, 3])
        step += 1
    
    return pd.DataFrame(episode_data)


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("📈 DQN-TFT Trading Bot Dashboard")
st.markdown("**Analyse comparative TFT vs LSTM** | Backtest + Live Trading")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Sélection mode
mode = st.sidebar.radio(
    "Mode de trading",
    ['📊 Backtest (Historique)', '🔴 Live Trading', '📈 Comparaison modèles']
)

# Sélection symbole
symbol = st.sidebar.selectbox(
    "Symbole boursier",
    ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
)

# Sélection modèle
if mode != '📈 Comparaison modèles':
    model_type = st.sidebar.radio(
        "Modèle à utiliser",
        ['TFT', 'LSTM']
    )

# Chemins fichiers
csv_path = st.sidebar.text_input(
    "Chemin CSV données",
    "dataset/top10_stocks_2025.csv"
)

lstm_model_path = st.sidebar.text_input(
    "Modèle LSTM",
    "newDQN/models/backup_models/dqn_trading_model_final.h5"
)

tft_model_path = st.sidebar.text_input(
    "Modèle TFT",
    "newDQN/models/backup_models/dqn_tft_trading_model_final.h5"
)

# ============================================================================
# MODE BACKTEST
# ============================================================================

if mode == '📊 Backtest (Historique)':
    st.header("📊 Backtest sur données historiques")
    
    # Chargement données
    data = load_stock_data(csv_path, symbol)
    
    if data is not None:
        st.success(f"✅ Données chargées: {len(data)} jours pour {symbol}")
        
        # Chargement agent
        model_path = tft_model_path if model_type == 'TFT' else lstm_model_path
        agent = load_trained_agent(model_type, model_path)
        
        if agent is not None and st.button("🚀 Lancer le backtest"):
            with st.spinner(f"Exécution backtest {model_type}..."):
                results = run_backtest(data, agent)
                
                # Graphique principal
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Prix et Actions', 'Portfolio Value', 'Récompenses'),
                    vertical_spacing=0.08,
                    row_heights=[0.4, 0.3, 0.3]
                )
                
                # Prix + actions
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['prices'],
                    mode='lines',
                    name='Prix',
                    line=dict(color='blue', width=2)
                ), row=1, col=1)
                
                # Marqueurs BUY/SELL
                buy_mask = results['actions'] == 'BUY'
                sell_mask = results['actions'] == 'SELL'
                
                fig.add_trace(go.Scatter(
                    x=results[buy_mask]['dates'],
                    y=results[buy_mask]['prices'],
                    mode='markers',
                    name='BUY',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=results[sell_mask]['dates'],
                    y=results[sell_mask]['prices'],
                    mode='markers',
                    name='SELL',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ), row=1, col=1)
                
                # Net Worth
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['net_worth'],
                    mode='lines',
                    name='Net Worth',
                    line=dict(color='purple', width=2),
                    fill='tozeroy'
                ), row=2, col=1)
                
                # Récompenses cumulées
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['rewards'].cumsum(),
                    mode='lines',
                    name='Rewards cumulés',
                    line=dict(color='orange', width=2)
                ), row=3, col=1)
                
                fig.update_layout(height=900, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques de performance
                st.subheader("📊 Métriques de performance")
                col1, col2, col3, col4 = st.columns(4)
                
                initial_value = 10000
                final_value = results['net_worth'].iloc[-1]
                total_return = ((final_value - initial_value) / initial_value) * 100
                
                col1.metric("Capital initial", f"${initial_value:.2f}")
                col2.metric("Capital final", f"${final_value:.2f}", f"{total_return:+.2f}%")
                col3.metric("Reward total", f"{results['rewards'].sum():.2f}")
                col4.metric("Nb trades", f"{(results['actions'] != 'HOLD').sum()}")
                
                # Distribution actions
                st.subheader("📈 Distribution des actions")
                action_counts = results['actions'].value_counts()
                fig_actions = go.Figure(data=[go.Pie(
                    labels=action_counts.index,
                    values=action_counts.values,
                    marker=dict(colors=['gray', 'green', 'red'])
                )])
                fig_actions.update_layout(height=400)
                st.plotly_chart(fig_actions, use_container_width=True)

# ============================================================================
# MODE LIVE
# ============================================================================

elif mode == '🔴 Live Trading':
    st.header("🔴 Trading en temps réel")
    st.warning("⚠️ Mode expérimental - Nécessite connexion Internet et yfinance")
    
    # Paramètres live
    duration = st.sidebar.slider("Durée simulation (minutes)", 1, 30, 5)
    
    # Chargement agent
    model_path = tft_model_path if model_type == 'TFT' else lstm_model_path
    agent = load_trained_agent(model_type, model_path)
    
    if agent is not None:
        if st.button(f"🚀 Lancer trading live {model_type} sur {symbol}"):
            live_data = run_live_simulation(symbol, agent, duration)
            
            if live_data is not None:
                st.success("✅ Simulation terminée")
                
                # Résumé
                st.subheader("📊 Résumé de la session")
                col1, col2, col3 = st.columns(3)
                
                initial_nw = live_data['net_worth'].iloc[0]
                final_nw = live_data['net_worth'].iloc[-1]
                pnl = final_nw - initial_nw
                
                col1.metric("Net Worth initial", f"${initial_nw:.2f}")
                col2.metric("Net Worth final", f"${final_nw:.2f}")
                col3.metric("P&L", f"${pnl:.2f}", f"{(pnl/initial_nw)*100:+.2f}%")
    else:
        st.error("❌ Impossible de charger l'agent. Vérifiez le chemin du modèle.")

# ============================================================================
# MODE COMPARAISON
# ============================================================================

elif mode == '📈 Comparaison modèles':
    st.header("📈 Comparaison TFT vs LSTM")
    
    # Chargement données
    data = load_stock_data(csv_path, symbol)
    
    if data is not None:
        # Chargement des deux agents
        lstm_agent = load_trained_agent('LSTM', lstm_model_path)
        tft_agent = load_trained_agent('TFT', tft_model_path)
        
        if lstm_agent is not None and tft_agent is not None:
            if st.button("🔬 Comparer les modèles"):
                with st.spinner("Exécution des backtests..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔵 LSTM")
                        lstm_results = run_backtest(data, lstm_agent)
                    
                    with col2:
                        st.subheader("🔴 TFT")
                        tft_results = run_backtest(data, tft_agent)
                    
                    # Graphique comparatif
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=lstm_results['dates'],
                        y=lstm_results['net_worth'],
                        mode='lines',
                        name='LSTM',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=tft_results['dates'],
                        y=tft_results['net_worth'],
                        mode='lines',
                        name='TFT',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Comparaison Net Worth",
                        xaxis_title="Date",
                        yaxis_title="Net Worth ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau comparatif
                    st.subheader("📊 Tableau comparatif")
                    
                    comparison = pd.DataFrame({
                        'Métrique': ['Capital final', 'Reward total', 'Nb trades', 'Win rate'],
                        'LSTM': [
                            f"${lstm_results['net_worth'].iloc[-1]:.2f}",
                            f"{lstm_results['rewards'].sum():.2f}",
                            f"{(lstm_results['actions'] != 'HOLD').sum()}",
                            f"{((lstm_results['rewards'] > 0).sum() / len(lstm_results) * 100):.1f}%"
                        ],
                        'TFT': [
                            f"${tft_results['net_worth'].iloc[-1]:.2f}",
                            f"{tft_results['rewards'].sum():.2f}",
                            f"{(tft_results['actions'] != 'HOLD').sum()}",
                            f"{((tft_results['rewards'] > 0).sum() / len(tft_results) * 100):.1f}%"
                        ]
                    })
                    
                    st.dataframe(comparison, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("**Projet IA Bourse - IPSSI MIA5** | Développé par Notre équipe de rêve | Octobre 2025")