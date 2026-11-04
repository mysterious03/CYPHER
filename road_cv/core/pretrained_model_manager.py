"""
Pretrained Model Manager for CYPhER.
Coordinates registry-driven loading, sequential GPU memory management,
checksum verification, and modular inference across pretrained model adapters.
"""
import os
import sys
import time
import importlib
import yaml
import hashlib
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union

class PretrainedModelManager:
    """
    Central orchestrator for pretrained AI road perception models.
    """
    def __init__(
        self,
        registry_path: str = "configs/pretrained_model_registry.yaml",
        config_path: str = "configs/pretrained_models.yaml",
        device: str = "cuda"
    ):
        self.registry_path = registry_path
        self.config_path = config_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.registry = self._load_yaml(registry_path).get("models", {})
        self.config = self._load_yaml(config_path)
        self.loaded_adapters: Dict[str, Any] = {}

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[PretrainedModelManager] Warning: failed loading {path}: {e}")
        return {}

    def verify_checksum(self, model_key: str) -> bool:
        """Verify SHA-256 checksum against registry."""
        entry = self.registry.get(model_key, {})
        ckpt = entry.get("checkpoint")
        expected_sha = entry.get("sha256")
        if not ckpt or not os.path.exists(ckpt) or not expected_sha:
            return False

        h = hashlib.sha256()
        with open(ckpt, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest().lower() == expected_sha.lower()

    def get_adapter(self, model_key: str):
        """
        Instantiate or retrieve adapter for a specific model key.
        Applies sequential loading and frees inactive GPU memory when appropriate.
        """
        if model_key in self.loaded_adapters:
            return self.loaded_adapters[model_key]

        entry = self.registry.get(model_key)
        if not entry:
            raise KeyError(f"Unknown model key in registry: {model_key}")

        adapter_class_path = entry.get("adapter_class")
        if not adapter_class_path:
            raise ValueError(f"No adapter_class specified for {model_key}")

        # Dynamic import
        module_name, class_name = adapter_class_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        adapter_cls = getattr(mod, class_name)

        # Sequential loading VRAM management: if GPU memory high, free previous adapters
        if torch.cuda.is_available() and self.config.get("memory_management", {}).get("sequential_loading", True):
            mem_used_gb = torch.cuda.memory_allocated() / (1024**3)
            target_env = self.config.get("memory_management", {}).get("target_vram_envelope_gb", 4.5)
            if mem_used_gb > target_env:
                self.release_all()

        adapter_instance = adapter_cls(device=self.device)
        self.loaded_adapters[model_key] = adapter_instance
        return adapter_instance

    def run_inference(self, model_key: str, image: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Execute inference with automatic error handling, latency, and VRAM logging.
        """
        adapter = self.get_adapter(model_key)

        # Track VRAM before and after
        vram_start_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        t0 = time.time()

        try:
            res = adapter.predict(image, **kwargs)
        except Exception as e:
            res = {
                "model_key": model_key,
                "status": "INFERENCE_ERROR",
                "error": str(e)
            }

        dt = time.time() - t0
        vram_end_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        res["manager_runtime_ms"] = round(dt * 1000, 2)
        res["vram_allocated_mb"] = round(vram_end_mb, 2)
        res["vram_delta_mb"] = round(vram_end_mb - vram_start_mb, 2)
        return res

    def run_profile(self, profile_name: str, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Execute a complete pipeline profile (FAST, BALANCED, HIGH_ACCURACY).
        """
        profile = self.config.get("profiles", {}).get(profile_name.upper())
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(self.config.get('profiles', {}).keys())}")

        results = {}
        total_time_ms = 0.0
        peak_vram_mb = 0.0

        for model_key in profile.get("models", []):
            entry = self.registry.get(model_key, {})
            # Skip unavailable or conflict checkpoints
            if entry.get("status") in ["CHECKPOINT_UNAVAILABLE", "DEPENDENCY_CONFLICT"]:
                results[model_key] = {
                    "status": entry.get("status"),
                    "note": entry.get("note", "Model skipped due to configuration")
                }
                continue

            res = self.run_inference(model_key, image)
            results[model_key] = res
            total_time_ms += res.get("manager_runtime_ms", 0.0)
            if torch.cuda.is_available():
                peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024**2))

            # If sequential loading is set, release adapter memory after each execution
            if self.config.get("memory_management", {}).get("sequential_loading", True):
                if hasattr(adapter := self.loaded_adapters.get(model_key), "release_resources"):
                    adapter.release_resources()
                if model_key in self.loaded_adapters:
                    del self.loaded_adapters[model_key]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return {
            "profile": profile_name,
            "models_executed": list(results.keys()),
            "total_latency_ms": round(total_time_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "results": results
        }

    def release_all(self):
        """Release all loaded models and free CUDA caches."""
        for name, adapter in list(self.loaded_adapters.items()):
            if hasattr(adapter, "release_resources"):
                adapter.release_resources()
        self.loaded_adapters.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_status_overview(self) -> Dict[str, Any]:
        """Summarize current registry status across all 11 model categories."""
        summary = {}
        for k, v in self.registry.items():
            summary[k] = {
                "task": v.get("task"),
                "architecture": v.get("architecture"),
                "status": v.get("status"),
                "license": v.get("license"),
                "checkpoint": v.get("checkpoint") or v.get("model_dir")
            }
        return summary

