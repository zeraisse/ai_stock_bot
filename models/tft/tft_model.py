"""
Modele TFT (Temporal Fusion Transformer) pour la prediction boursiere.

Je cree ce module pour wrapper le TFT de pytorch-forecasting et l'adapter
a mes besoins de trading. TFT combine plusieurs composants avances :
- Variable Selection Network (VSN) : selectionne les features importantes
- LSTM Encoder/Decoder : traite les sequences temporelles
- Multi-Head Attention : capture les dependances long-terme
- Gated Residual Network (GRN) : fusionne les informations
"""

import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
import pytorch_lightning as pl
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class TFTModel:
    """
    Je cree cette classe pour encapsuler le TFT de pytorch-forecasting.
    
    Le TFT est complexe avec plusieurs hyper-parametres. Je les regroupe ici
    pour faciliter l'utilisation et permettre le tuning.
    """
    
    def __init__(self,
                 hidden_size: int = 64,
                 lstm_layers: int = 2,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 16,
                 learning_rate: float = 0.001,
                 reduce_on_plateau_patience: int = 4):
        """
        J'initialise le modele TFT avec ses hyper-parametres.
        
        Args:
            hidden_size: Taille des couches cachees (64 est un bon compromis)
            lstm_layers: Nombre de couches LSTM (2 par defaut)
            attention_head_size: Nombre de tetes d'attention (4 par defaut)
            dropout: Taux de dropout pour regularisation (0.1 = 10%)
            hidden_continuous_size: Taille des embeddings pour variables continues
            learning_rate: Taux d'apprentissage
            reduce_on_plateau_patience: Patience avant reduction du learning rate
        """
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        
        # Le modele sera cree lors de l'appel a build_model()
        self.model = None
        
        print(f"Configuration TFT initialisee :")
        print(f"  - Hidden size : {hidden_size}")
        print(f"  - LSTM layers : {lstm_layers}")
        print(f"  - Attention heads : {attention_head_size}")
        print(f"  - Dropout : {dropout}")
    
    def build_model(self, training_dataset) -> TemporalFusionTransformer:
        """
        Je construis le modele TFT a partir du dataset d'entrainement.
        
        TFT a besoin de connaitre la structure des donnees (nombre de features,
        tailles des embeddings, etc.) donc je le cree a partir du dataset.
        
        Args:
            training_dataset: TimeSeriesDataSet d'entrainement
            
        Returns:
            model: Modele TFT configure
        """
        print("\nJe construis le modele TFT...")
        
        # Je cree le modele TFT
        # C'est ici que la magie opere : TFT va automatiquement configurer
        # toute son architecture complexe basee sur le dataset
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            
            # Architecture du reseau
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            
            # Fonction de perte
            # Je utilise QuantileLoss pour avoir des intervalles de confiance
            # au lieu d'une prediction ponctuelle
            loss=QuantileLoss(),
            
            # Optimiseur
            learning_rate=self.learning_rate,
            
            # Reduction du learning rate si pas d'amelioration
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
            
            # Je log les metriques pour suivre l'entrainement
            log_interval=10,
            log_val_interval=1,
        )
        
        # Je compte le nombre de parametres
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Modele TFT cree avec {num_params:,} parametres entrainables")
        
        return self.model
    
    def get_model_info(self) -> Dict:
        """
        Je retourne les informations sur le modele.
        
        Returns:
            dict: Dictionnaire avec les infos du modele
        """
        if self.model is None:
            return {"status": "Modele non encore cree"}
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "architecture": "Temporal Fusion Transformer",
            "hidden_size": self.hidden_size,
            "lstm_layers": self.lstm_layers,
            "attention_heads": self.attention_head_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "num_parameters": num_params,
            "components": [
                "Variable Selection Network (VSN)",
                "Static Covariate Encoder",
                "LSTM Encoder/Decoder",
                "Multi-Head Attention",
                "Gated Residual Network (GRN)",
                "Quantile Output Layer"
            ]
        }
    
    def explain_architecture(self):
        """
        J'explique l'architecture TFT composant par composant.
        """
        print("\n" + "="*70)
        print("ARCHITECTURE TFT (Temporal Fusion Transformer)")
        print("="*70)
        
        print("\n1. VARIABLE SELECTION NETWORK (VSN)")
        print("   Role : Je selectionne automatiquement les features importantes")
        print("   Comment : Gated Residual Network + Softmax sur chaque variable")
        print("   Avantage : Je reduis le bruit et me concentre sur l'essentiel")
        
        print("\n2. STATIC COVARIATE ENCODER")
        print("   Role : J'encode les variables statiques (ex: symbole boursier)")
        print("   Comment : Embeddings + GRN")
        print("   Avantage : Je capture les caracteristiques propres a chaque action")
        
        print("\n3. LSTM ENCODER/DECODER")
        print("   Role : Je traite les sequences temporelles")
        print("   Comment : LSTM bidirectionnel pour encoder + LSTM pour decoder")
        print("   Avantage : Je capture les patterns temporels courts et moyens")
        
        print("\n4. MULTI-HEAD ATTENTION")
        print("   Role : Je capture les dependances long-terme")
        print("   Comment : Self-attention avec plusieurs tetes")
        print("   Avantage : Je vois les relations entre periodes eloignees")
        
        print("\n5. GATED RESIDUAL NETWORK (GRN)")
        print("   Role : Je fusionne les informations de tous les composants")
        print("   Comment : Skip connections + gates pour controler le flux")
        print("   Avantage : Je stabilise l'entrainement et evite le gradient vanishing")
        
        print("\n6. QUANTILE OUTPUT LAYER")
        print("   Role : Je produis des predictions avec intervalles de confiance")
        print("   Comment : 3 quantiles (10%, 50%, 90%)")
        print("   Avantage : Je donne une mesure d'incertitude en plus de la prediction")
        
        print("\n" + "="*70)
        
        if self.model is not None:
            info = self.get_model_info()
            print(f"\nMon modele actuel a {info['num_parameters']:,} parametres")
            print(f"Hidden size: {info['hidden_size']}, LSTM layers: {info['lstm_layers']}")
            print(f"Attention heads: {info['attention_heads']}, Dropout: {info['dropout']}")
        
        print("="*70 + "\n")


class TFTTrainer:
    """
    Je cree cette classe pour faciliter l'entrainement du TFT.
    
    PyTorch Lightning gere beaucoup de choses automatiquement (GPU, logging,
    checkpointing, etc.) mais je veux garder un controle simple.
    """
    
    def __init__(self, 
                 max_epochs: int = 50,
                 gpus: int = 1 if torch.cuda.is_available() else 0,
                 gradient_clip_val: float = 0.1):
        """
        J'initialise le trainer PyTorch Lightning.
        
        Args:
            max_epochs: Nombre maximum d'epoques
            gpus: Nombre de GPUs (1 si disponible, sinon 0 pour CPU)
            gradient_clip_val: Valeur de clipping pour les gradients
        """
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.gradient_clip_val = gradient_clip_val
        
        device = "GPU" if gpus > 0 else "CPU"
        print(f"Trainer initialise pour {max_epochs} epochs sur {device}")
    
    def create_trainer(self, checkpoint_path: Optional[str] = None) -> pl.Trainer:
        """
        Je cree le trainer PyTorch Lightning.
        
        Args:
            checkpoint_path: Chemin pour sauvegarder les checkpoints
            
        Returns:
            trainer: Trainer PyTorch Lightning
        """
        # Je configure les callbacks
        callbacks = []
        
        # Early stopping : j'arrete si pas d'amelioration pendant 10 epochs
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode="min"
        )
        callbacks.append(early_stop_callback)
        
        # Model checkpoint : je sauvegarde le meilleur modele
        if checkpoint_path:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_path,
                filename='tft-{epoch:02d}-{val_loss:.4f}',
                monitor="val_loss",
                mode="min",
                save_top_k=3  # Je garde les 3 meilleurs modeles
            )
            callbacks.append(checkpoint_callback)
        
        # Je cree le trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        return trainer


# Fonction de test
def test_model():
    """
    Je teste la creation du modele TFT.
    """
    print("Test du modele TFT...")
    print("="*60)
    
    # Je cree un modele avec des parametres par defaut
    tft_model = TFTModel(
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4
    )
    
    # J'explique l'architecture
    tft_model.explain_architecture()
    
    # J'affiche les infos
    info = tft_model.get_model_info()
    print("\nInformations du modele :")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nTest reussi!")
    
    return tft_model


if __name__ == "__main__":
    test_model()