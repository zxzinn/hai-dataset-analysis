"""
Model implementations for anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for anomaly detection"""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        encoded, _ = self.encoder(x)
        
        # Decode
        decoded, _ = self.decoder(encoded)
        
        return decoded
    
    def get_reconstruction_error(self,
                               x: torch.Tensor,
                               batch_size: int = 32) -> np.ndarray:
        """Calculate reconstruction error for input sequences"""
        self.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                reconstructed = self(batch)
                error = torch.mean((batch - reconstructed) ** 2, dim=(1,2))
                errors.append(error.numpy())
                
        return np.concatenate(errors)

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(self,
                 input_size: int,
                 num_channels: List[int],
                 kernel_size: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size-1) * dilation,
                    dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, sequence, features]
        # Reshape for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply TCN
        x = self.network(x)
        
        # Apply final linear layer
        x = self.linear(x.transpose(1, 2))
        
        return x

class TemporalBlock(nn.Module):
    """Temporal Block for TCN"""
    
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)))
        return out

class ModelTrainer:
    """Trainer class for anomaly detection models"""
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output, batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self,
                val_loader: torch.utils.data.DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, batch)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self,
             train_loader: torch.utils.data.DataLoader,
             val_loader: Optional[torch.utils.data.DataLoader] = None,
             n_epochs: int = 50,
             patience: int = 5,
             model_path: Optional[str] = None) -> Dict[str, List[float]]:
        """Train model with early stopping"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if model_path is not None:
                        torch.save(self.model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
                
                print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}')
                
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

def prepare_sequences(data: np.ndarray,
                     sequence_length: int,
                     step: int = 1) -> Tuple[np.ndarray, StandardScaler]:
    """Prepare sequences for training"""
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    sequences = []
    for i in range(0, len(scaled_data) - sequence_length + 1, step):
        sequences.append(scaled_data[i:i+sequence_length])
    
    return np.array(sequences), scaler

def save_model(model: nn.Module,
              scaler: StandardScaler,
              save_dir: Path,
              model_name: str) -> None:
    """Save model and scaler"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), save_dir / f'{model_name}.pth')
    
    # Save scaler
    joblib.dump(scaler, save_dir / f'{model_name}_scaler.joblib')

def load_model(model: nn.Module,
              save_dir: Path,
              model_name: str) -> Tuple[nn.Module, StandardScaler]:
    """Load model and scaler"""
    # Load model
    model.load_state_dict(torch.load(save_dir / f'{model_name}.pth'))
    
    # Load scaler
    scaler = joblib.load(save_dir / f'{model_name}_scaler.joblib')
    
    return model, scaler
