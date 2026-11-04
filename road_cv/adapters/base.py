"""
CYPhER Base Model Adapter Interface
Establishes consistent input/output schemas, error boundaries, and VRAM management.
"""

import time
import abc
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch

class BaseAdapter(abc.ABC):
    """
    Standard Base Adapter for all CYPhER perception capabilities.
    """

    def __init__(self, mode: str, name: str, config: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.name = name
        self.config = config or {}
        self.status = "INITIALIZED"
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def load(self, device: Optional[str] = None) -> bool:
        """Loads model weights onto target device."""
        pass

    @abc.abstractmethod
    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Runs inference on a single BGR image frame.
        Returns a standardized dictionary matching CYPhER schema.
        """
        pass

    def unload(self):
        """Releases model from memory to safeguard 6 GB VRAM envelope."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def format_output(
        self,
        status: str,
        predictions: Any,
        warnings: Optional[List[str]] = None,
        latency_ms: Optional[float] = None,
        **extra
    ) -> Dict[str, Any]:
        """Formats output into standardized JSON-compatible dictionary."""
        out = {
            "mode": self.mode,
            "status": status,
            "model_name": self.name,
            "predictions": predictions,
            "warnings": warnings or [],
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None
        }
        out.update(extra)
        return out
