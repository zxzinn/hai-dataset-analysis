import torch
import torch.nn as nn
import numpy as np

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for anomaly detection in time series data
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # Encoder: input -> hidden state
        _, (hidden, cell) = self.encoder(x)
        
        # Create decoder input (hidden state repeated for each time step)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decoder: hidden state -> reconstruction
        output, _ = self.decoder(decoder_input, (hidden, cell))
        
        return output


class CausalConv1d(nn.Module):
    """
    1D causal convolution layer for TCN
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # Remove padding to maintain causality


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        return x + residual


class TCNAutoencoder(nn.Module):
    """
    Temporal Convolutional Network based autoencoder for anomaly detection
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], kernel_size=3, dropout=0.2):
        super(TCNAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_channels = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            dilation = 2 ** i
            encoder_layers.append(TCNBlock(in_channels, hidden_dim, kernel_size, dilation, dropout))
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (reverse of encoder)
        decoder_layers = []
        hidden_dims = hidden_dims[::-1]  # Reverse
        for i in range(len(hidden_dims)-1):
            dilation = 2 ** (len(hidden_dims) - i - 2)
            decoder_layers.append(TCNBlock(hidden_dims[i], hidden_dims[i+1], kernel_size, dilation, dropout))
        decoder_layers.append(nn.Conv1d(hidden_dims[-1], input_dim, 1))  # Final projection
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Input shape: [batch, seq_len, features]
        # Conv1d expects: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Return to original shape
        return decoded.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for anomaly detection
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(self, src):
        # Input projection
        src = self.input_projection(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoder
        memory = self.transformer_encoder(src)
        
        # Transformer decoder (using memory as both source and target)
        output = self.transformer_decoder(src, memory)
        
        # Output projection
        output = self.output_projection(output)
        
        return output