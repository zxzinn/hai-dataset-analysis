import json
import joblib
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

class ModelManager:
    """Model management utilities for HAI Security Dataset"""
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize model manager
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_model_dir(self, model_name: str, version: str) -> Path:
        """Get directory path for specific model version"""
        return self.base_dir / model_name / version
    
    def save_sklearn_model(self,
                          model: Any,
                          model_name: str,
                          version: str,
                          metadata: Dict[str, Any]) -> None:
        """
        Save scikit-learn model
        
        Args:
            model: Trained scikit-learn model
            model_name: Name of the model
            version: Version string
            metadata: Model metadata
        """
        model_dir = self._get_model_dir(model_name, version)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, model_dir / "model.joblib")
        
        # Save metadata
        self._save_metadata(model_dir, metadata)
    
    def load_sklearn_model(self,
                          model_name: str,
                          version: str) -> tuple[Any, Dict[str, Any]]:
        """
        Load scikit-learn model
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Tuple of (model, metadata)
        """
        model_dir = self._get_model_dir(model_name, version)
        
        # Load model
        model = joblib.load(model_dir / "model.joblib")
        
        # Load metadata
        metadata = self._load_metadata(model_dir)
        
        return model, metadata
    
    def save_torch_model(self,
                        model: torch.nn.Module,
                        model_name: str,
                        version: str,
                        metadata: Dict[str, Any]) -> None:
        """
        Save PyTorch model
        
        Args:
            model: Trained PyTorch model
            model_name: Name of the model
            version: Version string
            metadata: Model metadata
        """
        model_dir = self._get_model_dir(model_name, version)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model architecture
        architecture = {
            'class_name': model.__class__.__name__,
            'config': model.state_dict()
        }
        torch.save(architecture, model_dir / "architecture.pth")
        
        # Save model weights
        torch.save(model.state_dict(), model_dir / "weights.pth")
        
        # Save metadata
        self._save_metadata(model_dir, metadata)
    
    def load_torch_model(self,
                        model_class: type,
                        model_name: str,
                        version: str) -> tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load PyTorch model
        
        Args:
            model_class: Model class to instantiate
            model_name: Name of the model
            version: Version string
            
        Returns:
            Tuple of (model, metadata)
        """
        model_dir = self._get_model_dir(model_name, version)
        
        # Load architecture
        architecture = torch.load(model_dir / "architecture.pth")
        
        # Create model instance
        model = model_class()
        
        # Load weights
        model.load_state_dict(torch.load(model_dir / "weights.pth"))
        
        # Load metadata
        metadata = self._load_metadata(model_dir)
        
        return model, metadata
    
    def _save_metadata(self, model_dir: Path, metadata: Dict[str, Any]) -> None:
        """Save model metadata"""
        # Add timestamp
        metadata['timestamp'] = datetime.now().isoformat()
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def _load_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """Load model metadata"""
        with open(model_dir / "metadata.json", 'r') as f:
            return json.load(f)
    
    def save_training_history(self,
                            history: Dict[str, list],
                            model_name: str,
                            version: str) -> None:
        """
        Save training history
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
            version: Version string
        """
        model_dir = self._get_model_dir(model_name, version)
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                history_json[key] = value.tolist()
            else:
                history_json[key] = value
                
        with open(model_dir / "history.json", 'w') as f:
            json.dump(history_json, f, indent=4)
    
    def load_training_history(self,
                            model_name: str,
                            version: str) -> Dict[str, list]:
        """
        Load training history
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Training history dictionary
        """
        model_dir = self._get_model_dir(model_name, version)
        
        with open(model_dir / "history.json", 'r') as f:
            return json.load(f)
    
    def list_versions(self, model_name: str) -> list[str]:
        """
        List available versions for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            return []
            
        return [d.name for d in model_dir.iterdir() if d.is_dir()]
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get latest version of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(model_name)
        if not versions:
            return None
            
        return sorted(versions)[-1]
