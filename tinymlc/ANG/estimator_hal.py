# estimator_hal.py
# Hardware HAL estimator - user-provided script for real hardware.

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from tinymlc.ANG.estimator import Estimator
from tinymlc.ANG.utils import (calculate_macs, calculate_params,
                               calculate_peak_ram)


class HardwareHALEstimator(Estimator):
    """
    Hardware HAL estimator (closed-loop).

    This estimator calls a user-provided Python script that interfaces
    with real hardware. The script must implement the estimate() function.

    This is a "closed-loop" estimator because it provides feedback
    from actual hardware execution.

    Vendors can implement their own hardware testing scripts following
    the same interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hardware HAL estimator.

        Args:
            config: Must contain 'script_path' pointing to the user script.
        """
        super().__init__(config)

        self.default_config = {
            "script_path": None,  # Path to user script
            "function_name": "estimate",  # Function name in user script
            "timeout": 60,  # Timeout in seconds
            "max_macs": 100000,
            "max_params": 50000,
            "max_ram": 32768,
        }
        self.config = {**self.default_config, **(config or {})}
        self._estimator_func = None

        if self.config.get("script_path"):
            self._load_estimator()

    def _load_estimator(self) -> None:
        """
        Load the user-provided estimator function.
        """
        script_path = self.config.get("script_path")
        if not script_path:
            raise ValueError(
                "HardwareHALEstimator requires 'script_path' in config"
            )

        path = Path(script_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Estimator script not found: {script_path}"
            )

        # Dynamic import of the user script
        spec = importlib.util.spec_from_file_location("user_estimator", path)
        if spec is None:
            raise ImportError(f"Cannot load script: {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["user_estimator"] = module
        if spec.loader is None:
            raise ImportError(f"Cannot find loader for: {script_path}")
        spec.loader.exec_module(module)

        func_name = self.config.get("function_name", "estimate")
        if not hasattr(module, func_name):
            raise AttributeError(
                f"Function '{func_name}' not found in {script_path}"
            )

        self._estimator_func = getattr(module, func_name)

    def estimate(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate performance using the user-provided hardware script.

        Args:
            model_info: ModelInfo dictionary.

        Returns:
            Dictionary with performance metrics.
        """
        # Get software metrics as fallback
        macs = calculate_macs(model_info)
        params = calculate_params(model_info)
        peak_ram = calculate_peak_ram(model_info)

        if self._estimator_func is None:
            self._load_estimator()

        try:
            # Call the user's estimate function
            result = self._estimator_func(model_info)

            # Validate the result has all required fields
            required_keys = ["score", "macs", "params", "peak_ram", "flash"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(
                        f"Estimator result missing required key: {key}"
                    )

            # If latency_ms is not provided, estimate it
            if "latency_ms" not in result:
                clock = self.config.get("clock_speed", 100000000)
                result["latency_ms"] = (macs / clock) * 1000.0

            return result

        except Exception as e:
            # Fallback to software estimation on error
            max_macs = self.config["max_macs"]
            max_params = self.config["max_params"]
            max_ram = self.config["max_ram"]

            macs_score = (
                1.0 - min(macs / max_macs, 1.0) if max_macs > 0 else 0.0
            )
            params_score = (
                1.0 - min(params / max_params, 1.0) if max_params > 0 else 0.0
            )
            ram_score = (
                1.0 - min(peak_ram / max_ram, 1.0) if max_ram > 0 else 0.0
            )

            score = (
                0.4 * macs_score + 0.3 * params_score + 0.3 * ram_score
            ) * 100.0

            return {
                "score": score,
                "macs": macs,
                "params": params,
                "peak_ram": peak_ram,
                "flash": params + 1024,
                "latency_ms": (macs / 100000000) * 1000.0,
                "details": {
                    "estimator": "hardware_hal",
                    "fallback": True,
                    "error": str(e),
                },
            }

    def get_info(self) -> Dict[str, str]:
        """Get estimator information."""
        return {
            "name": "HardwareHALEstimator",
            "version": "1.0",
            "type": "closed_loop",
            "description": "Hardware HAL estimator with user script",
            "script_path": str(self.config.get("script_path", "")),
            "function_name": self.config.get("function_name", "estimate"),
        }
