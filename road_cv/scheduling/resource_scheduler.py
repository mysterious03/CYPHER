"""
Resource-Aware Inference Scheduling and VRAM Management Module
Safeguards execution within the 6 GB VRAM envelope on RTX 4050, tracks peak memory,
profiles stage latency, and adaptively budgets computation.
"""

import gc
import time
from typing import Dict, Any, Optional
import torch

class GPUResourceScheduler:
    """
    Monitors GPU memory, enforces sequential model unloading, and dynamically
    allocates crop inference budgets based on available hardware headroom.
    """

    def __init__(
        self,
        max_vram_gb: float = 6.0,
        safety_headroom_gb: float = 1.0, # Reserve 1 GB buffer to prevent OOM
        default_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.max_vram_gb = max_vram_gb
        self.safety_headroom_gb = safety_headroom_gb
        self.device = default_device
        self.reset_peak_memory()

    def reset_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_memory_status(self) -> Dict[str, float]:
        """
        Returns allocated, reserved, and peak GPU memory in Gigabytes.
        """
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "peak_allocated_gb": 0.0,
                "free_headroom_gb": self.max_vram_gb
            }

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        free_headroom = max(0.0, self.max_vram_gb - reserved)

        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "peak_allocated_gb": round(peak, 3),
            "free_headroom_gb": round(free_headroom, 3)
        }

    def allocate_crop_budget(self, requested_crops: int = 4) -> int:
        """
        Calculates safe maximum refinement crops given current VRAM headroom.
        """
        mem = self.get_memory_status()
        headroom = mem["free_headroom_gb"]

        if headroom > 2.5:
            return requested_crops
        elif headroom > 1.5:
            return min(requested_crops, 2)
        elif headroom > self.safety_headroom_gb:
            return min(requested_crops, 1)
        else:
            # Low memory: skip refinement to avert CUDA OOM
            return 0

    @staticmethod
    def clear_gpu_cache():
        """
        Forces garbage collection and releases PyTorch cached memory blocks.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LatencyProfiler:
    """
    Profiles component-level execution latency and throughput.
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, stage_name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_times[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_t = self._start_times.get(stage_name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.timings[stage_name] = round(elapsed_ms, 2)
        return self.timings[stage_name]

    def get_summary(self) -> Dict[str, Any]:
        total_latency = sum(self.timings.values())
        return {
            "stage_latencies_ms": dict(self.timings),
            "total_latency_ms": round(total_latency, 2),
            "estimated_fps": round(1000.0 / max(1.0, total_latency), 1)
        }

