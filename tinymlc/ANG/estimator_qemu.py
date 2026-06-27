# estimator_qemu.py
# QEMU-based estimator using icount mode for deterministic instruction counts.

import subprocess
import tempfile
import re
from typing import Dict, Any, Optional

from tinymlc.ANG.estimator import Estimator
from tinymlc.ANG.utils import (calculate_macs, calculate_params,
                               calculate_peak_ram)


class QemuEstimator(Estimator):
    """
    QEMU-based estimator (closed-loop).

    This estimator compiles the model to an ELF file and runs it under
    QEMU with icount mode enabled. The instruction count is used as
    a stable, deterministic performance metric.

    This is a "closed-loop" estimator because it provides feedback
    from actual execution (even if simulated).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QEMU estimator.

        Args:
            config: Configuration with QEMU and compilation settings.
        """
        super().__init__(config)

        self.default_config = {
            "qemu_binary": "qemu-system-arm",  # QEMU binary to use
            "cpu": "cortex-m4",  # CPU model
            "icount_shift": 0,  # icount shift (0 = 1 instr/tick)
            "clock_speed": 100000000,  # Clock speed in Hz
            "timeout": 30,  # Timeout in seconds
            "max_macs": 100000,  # For score calculation
            "max_params": 50000,
            "max_ram": 32768,
            "gcc_binary": "arm-none-eabi-gcc",  # Cross-compiler
            "linker_script": "mcu.ld",  # Linker script path
        }
        self.config = {**self.default_config, **(config or {})}
        self._compile_cache = {}

    def estimate(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate performance by running the model under QEMU.

        Args:
            model_info: ModelInfo dictionary.

        Returns:
            Dictionary with performance metrics.
        """
        # Get software metrics first
        macs = calculate_macs(model_info)
        params = calculate_params(model_info)
        peak_ram = calculate_peak_ram(model_info)

        # Compile and run under QEMU
        try:
            elf_path = self._compile_model(model_info)
            instr_count = self._run_qemu_icount(elf_path)
        except Exception as e:
            # On failure, fall back to software estimates
            instr_count = macs  # Approximate instruction count

        # Estimate latency from instruction count
        latency_ms = (instr_count / self.config["clock_speed"]) * 1000.0

        # Flash usage from ELF size
        flash = (
            self._get_elf_flash_size(elf_path)
            if elf_path
            else params + 1024
        )

        # Calculate score (same as software estimator)
        max_macs = self.config["max_macs"]
        max_params = self.config["max_params"]
        max_ram = self.config["max_ram"]

        macs_score = 1.0 - min(macs / max_macs, 1.0) if max_macs > 0 else 0.0
        params_score = (
            1.0 - min(params / max_params, 1.0) if max_params > 0 else 0.0
        )
        ram_score = 1.0 - min(peak_ram / max_ram, 1.0) if max_ram > 0 else 0.0

        score = (
            0.4 * macs_score + 0.3 * params_score + 0.3 * ram_score
        ) * 100.0

        return {
            "score": score,
            "macs": macs,
            "params": params,
            "peak_ram": peak_ram,
            "flash": flash,
            "latency_ms": latency_ms,
            "instr_count": instr_count,  # QEMU-specific
            "details": {
                "estimator": "qemu",
                "qemu_cpu": self.config["cpu"],
                "icount_shift": self.config["icount_shift"],
                "elf_path": elf_path,
            },
        }

    def get_info(self) -> Dict[str, str]:
        """Get estimator information."""
        return {
            "name": "QemuEstimator",
            "version": "1.0",
            "type": "closed_loop",
            "description": "QEMU icount-based estimator",
            "qemu_binary": self.config["qemu_binary"],
            "cpu": self.config["cpu"],
        }

    def _compile_model(self, model_info: Dict[str, Any]) -> str:
        """
        Compile the model to an ELF file.

        Args:
            model_info: ModelInfo dictionary.

        Returns:
            Path to the compiled ELF file.
        """
        # This would call the TinyMLC code generator
        # For now, we return a placeholder
        # In production, this would:
        #   1. Call generate_c_code(model_info)
        #   2. Compile with the cross-compiler
        #   3. Return the ELF path

        # Placeholder implementation
        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
            elf_path = f.name

        # TODO: Actually compile the model
        # This requires integration with TinyMLC code generation

        return elf_path

    def _run_qemu_icount(self, elf_path: str) -> int:
        """
        Run the ELF under QEMU with icount mode.

        Args:
            elf_path: Path to the ELF file.

        Returns:
            Instruction count.
        """
        cmd = [
            self.config["qemu_binary"],
            "-cpu",
            self.config["cpu"],
            "-nographic",
            "-icount",
            f"shift={self.config['icount_shift']}",
            "-semihosting",
            "-semihosting-config",
            "enable=on,target=native",
            "-kernel",
            elf_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["timeout"],
            )
            # Parse instruction count from output
            return self._parse_icount(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            return 0

    def _parse_icount(self, output: str) -> int:
        """
        Parse instruction count from QEMU output.

        Args:
            output: QEMU stdout/stderr.

        Returns:
            Instruction count.
        """
        # Try different patterns
        patterns = [
            r"INSTR_COUNT:\s*(\d+)",
            r"icount:\s*(\d+)",
            r"instructions\s*:\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 0

    def _get_elf_flash_size(self, elf_path: str) -> int:
        """
        Get flash size from ELF using size command.

        Args:
            elf_path: Path to the ELF file.

        Returns:
            Flash size in bytes.
        """
        try:
            # Use GNU size to get text + data
            size_cmd = self.config.get("size_binary", "arm-none-eabi-size")
            result = subprocess.run(
                [size_cmd, elf_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 3:
                        # text, data, bss
                        text = int(parts[0])
                        data = int(parts[1])
                        return text + data
        except Exception:
            pass
        return 0
