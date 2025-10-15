import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "UNH", "JNJ"
]

def format_for_excel(df):
    """
    Formate le DataFrame pour une meilleure compatibilité avec Excel
    """
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    numeric_columns = ['Open', 'High', 'Low', 'Close']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
    
    return df

def download_stock_data():
    """
    Télécharge les données boursières et les formate proprement
    """
    all_data = []
    
    print("=== Téléchargement des données boursières ===")
    
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] Téléchargement de {symbol}...")
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start="2020-01-01", end="2025-10-15")
            
            if data.empty:
                print(f"Aucune donnée disponible pour {symbol}")
                continue
            
            data = data.reset_index()
            
            data["Symbol"] = symbol
            
            data = data[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]
            
            data = format_for_excel(data)
            
            all_data.append(data)
            print(f"{symbol}: {len(data)} lignes récupérées")
            
        except Exception as e:
            print(f"Erreur pour {symbol}: {str(e)}")
            continue
    
    if not all_data:
        print("Aucune donnée récupérée!")
        return None
    
    print("\n=== Consolidation des données ===")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df = combined_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    
    return combined_df

def save_to_csv(df, filename="top10_stocks_2025_clean.csv"):
    """
    Sauvegarde le DataFrame dans un fichier CSV compatible Excel
    """
    try:
        df.to_csv(
            filename, 
            index=False,                    
            encoding='utf-8-sig',         
            sep=';',                       
            decimal=',',                   
            date_format='%Y-%m-%d'        
        )
        
        filename_int = filename.replace('.csv', '_international.csv')
        df.to_csv(
            filename_int,
            index=False,
            encoding='utf-8-sig',
            sep=',',                      
            decimal='.',                   
            date_format='%Y-%m-%d'
        )
        
        return filename, filename_int
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {str(e)}")
        return None, None

def main():
    """
    Fonction principale
    """
    print("Démarrage du téléchargement des données boursières\n")
    
    df = download_stock_data()
    
    if df is None:
        return
    
    print("\n=== Sauvegarde des fichiers ===")
    file_eu, file_int = save_to_csv(df)
    
    if file_eu and file_int:
        print(f"Fichier EU (séparateur ;): {file_eu}")
        print(f"Fichier International (séparateur ,): {file_int}")
        
        print(f"\n=== Statistiques ===")
        print(f"Nombre total de lignes: {len(df):,}")
        print(f"Symboles inclus: {', '.join(sorted(df['Symbol'].unique()))}")
        print(f"Période: du {df['Date'].min()} au {df['Date'].max()}")
        
        print(f"\n=== Aperçu des données ===")
        print(df.head(10).to_string(index=False))
        
    else:
        print("Erreur lors de la sauvegarde")

if __name__ == "__main__":
    main()