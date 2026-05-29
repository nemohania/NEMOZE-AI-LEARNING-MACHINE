"""
PyTorch-based Neural Network Models
"""

import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    """Transformer-based model for sequential data prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 8):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        # Embed input
        embedded = self.embedding(x)
        
        # Apply transformer
        transformed = self.transformer(embedded)
        
        # Output layer
        output = self.output_layer(transformed)
        
        return output


class LSTMModel(nn.Module):
    """LSTM-based model for sequence prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, output_dim: int = None):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or input_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: Output tensor, hidden state tuple
        """
        lstm_out, hidden = self.lstm(x)
        output = self.output_layer(lstm_out)
        return output, hidden


class CNNModel(nn.Module):
    """Convolutional Neural Network for feature extraction"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, input_channels, seq_len)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x


class ModelFactory:
    """Factory for creating different model types"""
    
    MODELS = {
        'transformer': TransformerModel,
        'lstm': LSTMModel,
        'cnn': CNNModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> nn.Module:
        """Create a model of the specified type"""
        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Creating {model_type} model with kwargs: {kwargs}")
        return cls.MODELS[model_type](**kwargs)
