# estimator_software.py
# Software-based estimator using MACs and parameter counts.

from typing import Dict, Any, Optional
from TinyMLC.ANG.estimator import Estimator
from TinyMLC.ANG.utils import (calculate_macs, calculate_params,
                               calculate_peak_ram)


class SoftwareEstimator(Estimator):
    """
    Pure software estimator (open-loop).

    This estimator uses mathematical formulas to compute:
        - MACs (multiply-accumulate operations)
        - Parameter count
        - Peak RAM usage (estimated)

    It does NOT require any hardware or simulation.

    This is a "open-loop" estimator because there is no feedback
    from actual hardware execution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the software estimator.

        Args:
            config: Configuration with max constraints and weighting.
        """
        super().__init__(config)

        self.default_config = {
            "max_macs": 100000,  # Maximum allowed MACs
            "max_params": 50000,  # Maximum allowed parameters
            "max_ram": 32768,  # Maximum RAM in bytes
            "weight_macs": 0.4,  # Weight for MACs in score
            "weight_params": 0.3,  # Weight for params in score
            "weight_ram": 0.3,  # Weight for RAM in score
            "clock_speed": 100000000,
            # Clock speed in Hz (for latency estimate)
        }
        # Merge with user config
        self.config = {**self.default_config, **(config or {})}

    def estimate(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate performance using pure software calculations.

        Args:
            model_info: ModelInfo dictionary.

        Returns:
            Dictionary with performance metrics.
        """
        macs = calculate_macs(model_info)
        params = calculate_params(model_info)
        peak_ram = calculate_peak_ram(model_info)

        # Estimate flash usage: params + overhead
        # Overhead is approximated as 1KB for code and metadata
        flash = params + 1024

        # Estimate latency: MACs / clock speed (ideal case)
        # This is a theoretical lower bound, not real-world latency
        latency_ms = (macs / self.config["clock_speed"]) * 1000.0

        # Calculate score (higher is better)
        # Normalize each metric to [0, 1] range
        max_macs = self.config["max_macs"]
        max_params = self.config["max_params"]
        max_ram = self.config["max_ram"]

        # Avoid division by zero
        macs_score = 1.0 - min(macs / max_macs, 1.0) if max_macs > 0 else 0.0
        params_score = (
            1.0 - min(params / max_params, 1.0) if max_params > 0 else 0.0
        )
        ram_score = 1.0 - min(peak_ram / max_ram, 1.0) if max_ram > 0 else 0.0

        # Weighted combination
        score = (
            self.config["weight_macs"] * macs_score
            + self.config["weight_params"] * params_score
            + self.config["weight_ram"] * ram_score
        ) * 100.0  # Scale to 0-100

        return {
            "score": score,
            "macs": macs,
            "params": params,
            "peak_ram": peak_ram,
            "flash": flash,
            "latency_ms": latency_ms,
            "details": {
                "estimator": "software",
                "macs_score": macs_score,
                "params_score": params_score,
                "ram_score": ram_score,
            },
        }

    def get_info(self) -> Dict[str, str]:
        """Get estimator information."""
        return {
            "name": "SoftwareEstimator",
            "version": "1.0",
            "type": "open_loop",
            "description": "Pure software estimator using MACs and params",
            "clock_speed": str(self.config["clock_speed"]),
        }
