"""
Architecture LSTM avec attention pour la prediction boursiere.

Je cree ce module pour definir l'architecture du reseau LSTM.
Cette architecture combine :
- Couches LSTM pour les sequences temporelles
- Mecanisme d'attention pour se concentrer sur les periodes importantes
- Couches fully connected pour les predictions finales
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Je cree ce modele LSTM avec attention pour predire les prix.
    
    Architecture :
    1. LSTM : Traite les sequences temporelles
    2. Attention : Pondere l'importance de chaque pas de temps
    3. Fully Connected : Produit les predictions finales
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,
                 dropout: float = 0.2):
        """
        J'initialise le modele LSTM.
        
        Args:
            input_size: Nombre de features en entree (12 avec indicateurs techniques)
            hidden_size: Taille des couches cachees (128 par defaut)
            num_layers: Nombre de couches LSTM empilees (2 par defaut)
            output_size: Nombre de predictions (3: prix, tendance, volatilite)
            dropout: Taux de dropout pour regularisation (0.2 = 20%)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # COUCHE 1 : LSTM
        # Je cree les couches LSTM pour traiter les sequences temporelles
        # batch_first=True signifie que le format d'entree est (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout entre les couches LSTM
        )
        
        # COUCHE 2 : ATTENTION
        # Je cree un mecanisme d'attention pour pondere l'importance de chaque pas de temps
        # L'attention va apprendre quels jours sont les plus importants pour la prediction
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Transformation des features LSTM
            nn.Tanh(),                            # Activation non-lineaire
            nn.Linear(hidden_size, 1)             # Score d'attention pour chaque pas de temps
        )
        
        # COUCHE 3 : FULLY CONNECTED
        # Je cree les couches finales pour produire les predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # Reduction de dimension
            nn.ReLU(),                                  # Activation
            nn.Dropout(dropout),                        # Regularisation
            nn.Linear(hidden_size // 2, output_size)   # Predictions finales
        )
        
        print(f"Modele LSTM cree avec :")
        print(f"  - Input size : {input_size}")
        print(f"  - Hidden size : {hidden_size}")
        print(f"  - Num layers : {num_layers}")
        print(f"  - Output size : {output_size}")
        print(f"  - Dropout : {dropout}")
    
    def forward(self, x):
        """
        Je realise la passe forward du modele.
        
        Etapes :
        1. LSTM traite la sequence et produit des hidden states
        2. Attention calcule des poids pour chaque pas de temps
        3. Je cree un contexte pondere par l'attention
        4. FC layers produisent les predictions finales
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_size)
               Par exemple : (32, 60, 12) = 32 sequences de 60 jours avec 12 features
            
        Returns:
            predictions: Tensor de forme (batch_size, output_size)
                        Par exemple : (32, 3) = 32 predictions avec 3 valeurs
        """
        # ETAPE 1 : LSTM
        # Je traite la sequence avec LSTM
        # lstm_out contient les hidden states pour chaque pas de temps
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape : (batch_size, sequence_length, hidden_size)
        # Par exemple : (32, 60, 128)
        
        # ETAPE 2 : ATTENTION
        # Je calcule les scores d'attention pour chaque pas de temps
        attention_scores = self.attention(lstm_out)
        # attention_scores shape : (batch_size, sequence_length, 1)
        # Par exemple : (32, 60, 1)
        
        # Je convertis les scores en poids avec softmax
        # Softmax assure que les poids somment a 1
        attention_weights = torch.softmax(attention_scores, dim=1)
        # attention_weights shape : (batch_size, sequence_length, 1)
        
        # ETAPE 3 : CONTEXTE PONDERE
        # Je cree un vecteur contexte en ponderant les hidden states par l'attention
        # Cela me permet de me concentrer sur les jours les plus importants
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context shape : (batch_size, hidden_size)
        # Par exemple : (32, 128)
        
        # ETAPE 4 : PREDICTIONS
        # Je passe le contexte dans les couches fully connected
        predictions = self.fc(context)
        # predictions shape : (batch_size, output_size)
        # Par exemple : (32, 3) = 32 predictions de [prix_ratio, tendance, volatilite]
        
        return predictions
    
    def get_attention_weights(self, x):
        """
        Je recupere les poids d'attention pour visualisation.
        
        Cette methode est utile pour analyser quels jours le modele
        considere comme les plus importants pour ses predictions.
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_size)
            
        Returns:
            attention_weights: Tensor de forme (batch_size, sequence_length, 1)
        """
        # Je fais juste la partie LSTM + Attention sans les predictions
        lstm_out, _ = self.lstm(x)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        return attention_weights


# Fonction utilitaire pour compter les parametres
def count_parameters(model):
    """
    Je compte le nombre de parametres entrainables du modele.
    
    Args:
        model: Modele PyTorch
        
    Returns:
        num_params: Nombre de parametres
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


# Test du modele
if __name__ == "__main__":
    print("Test du modele LSTM...")
    print("="*60)
    
    # Je cree un modele de test
    input_size = 12  # 12 features (OHLCV + indicateurs)
    hidden_size = 128
    num_layers = 2
    output_size = 3  # Prix, tendance, volatilite
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
    
    # Je compte les parametres
    num_params = count_parameters(model)
    print(f"\nNombre de parametres : {num_params:,}")
    
    # Je teste avec des donnees factices
    batch_size = 32
    sequence_length = 60
    
    # Je cree un batch d'exemple
    x = torch.randn(batch_size, sequence_length, input_size)
    print(f"\nInput shape : {x.shape}")
    
    # Je fais une passe forward
    predictions = model(x)
    print(f"Output shape : {predictions.shape}")
    
    # Je recupere les poids d'attention
    attention_weights = model.get_attention_weights(x)
    print(f"Attention weights shape : {attention_weights.shape}")
    
    print("\nTest reussi!")
    print("="*60)