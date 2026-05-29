"""
Model Manager for loading, saving, and managing ML models
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json

from app.config import settings
from app.models.neural_network import ModelFactory

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model lifecycle: creation, training, saving, loading"""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model = None
        self.model_config = {}
        
    def create_model(self, model_type: str, **kwargs) -> nn.Module:
        """Create a new model"""
        logger.info(f"Creating new {model_type} model")
        model = ModelFactory.create_model(model_type, **kwargs)
        model.to(self.device)
        self.current_model = model
        self.model_config = {
            'type': model_type,
            'kwargs': kwargs,
            'device': str(self.device)
        }
        return model
    
    def save_model(self, model: nn.Module, name: str, metadata: Dict[str, Any] = None) -> str:
        """Save model to disk"""
        model_path = self.models_dir / f"{name}.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': self.model_config.get('type', 'unknown'),
            'config': self.model_config,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, name: str) -> nn.Module:
        """Load model from disk"""
        model_path = self.models_dir / f"{name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'transformer')
        config = checkpoint.get('config', {})
        kwargs = config.get('kwargs', {})
        
        # Create model
        model = ModelFactory.create_model(model_type, **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.current_model = model
        self.model_config = config
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def get_model(self) -> Optional[nn.Module]:
        """Get current model"""
        return self.current_model
    
    def list_models(self) -> list:
        """List all saved models"""
        models = []
        for model_file in self.models_dir.glob("*.pt"):
            models.append(model_file.stem)
        return models
    
    def delete_model(self, name: str) -> bool:
        """Delete a model"""
        model_path = self.models_dir / f"{name}.pt"
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Model deleted: {name}")
            return True
        return False
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get model information"""
        model_path = self.models_dir / f"{name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        return {
            'name': name,
            'type': checkpoint.get('model_type', 'unknown'),
            'config': checkpoint.get('config', {}),
            'metadata': checkpoint.get('metadata', {}),
            'file_size': os.path.getsize(model_path),
        }
