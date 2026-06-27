# utils.py
# Common utility functions used across ANG.

import copy
import json
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np


def hash_structure(structure: Dict[str, Any]) -> str:
    """
    Compute a hash of a network structure (op sequence only, not weights).

    This is used for caching and duplicate detection.

    Args:
        structure: Dictionary containing the network structure.

    Returns:
        SHA256 hash string.
    """
    # Extract only the structure-defining parts
    ops = structure.get("ops", [])
    # Normalize to a stable representation
    normalized = []
    for op in ops:
        op_copy = {
            "op_name": op.get("op_name"),
            "params": op.get("params", {}),
        }
        normalized.append(op_copy)

    # Sort to ensure stability
    normalized.sort(key=lambda x: str(x))
    json_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def calculate_macs(model_info: Dict[str, Any]) -> int:
    """
    Calculate the total number of multiply-accumulate operations.

    Args:
        model_info: ModelInfo dictionary.

    Returns:
        Total MACs count.
    """
    total = 0
    ops = model_info.get("ops", [])
    tensors = model_info.get("tensors", {})

    for op in ops:
        op_name = op.get("op_name")

        if op_name == "CONV_2D":
            conv_params = op.get("conv_params", {})
            kernel_size = conv_params.get("kernel_size", 3)
            stride = conv_params.get("stride", 1)

            # Get input and output shapes
            input_idx = op.get("input_indices", [0])[0]
            output_idx = op.get("output_indices", [0])[0]

            input_shape = tensors.get(input_idx, {}).get(
                "shape", [1, 1, 1, 1]
            )
            output_shape = tensors.get(output_idx, {}).get(
                "shape", [1, 1, 1, 1]
            )

            # NHWC format: [batch, height, width, channels]
            if len(input_shape) >= 4 and len(output_shape) >= 4:
                h = output_shape[1] if output_shape[1] else 1
                w = output_shape[2] if output_shape[2] else 1
                c_in = input_shape[3] if input_shape[3] else 1
                c_out = output_shape[3] if output_shape[3] else 1
                # MACs = H * W * C_in * C_out * K * K
                # Bias is not counted as a MAC
                total += h * w * c_in * c_out * kernel_size * kernel_size

        elif op_name == "FULLY_CONNECTED":
            fc_params = op.get("fc_params", {})
            input_size = 1
            output_size = fc_params.get("units", 64)

            # Get input shape
            input_idx = op.get("input_indices", [0])[0]
            input_shape = tensors.get(input_idx, {}).get("shape", [])
            for dim in input_shape:
                input_size *= dim

            # MACs = input_size * output_size
            total += input_size * output_size

        elif op_name == "DEPTHWISE_CONV_2D":
            # Similar to CONV_2D but with groups = C_in = C_out
            dw_params = op.get("dw_params", {})
            kernel_size = dw_params.get("kernel_size", 3)

            input_idx = op.get("input_indices", [0])[0]
            output_idx = op.get("output_indices", [0])[0]

            input_shape = tensors.get(input_idx, {}).get(
                "shape", [1, 1, 1, 1]
            )
            output_shape = tensors.get(output_idx, {}).get(
                "shape", [1, 1, 1, 1]
            )

            if len(input_shape) >= 4 and len(output_shape) >= 4:
                h = output_shape[1] if output_shape[1] else 1
                w = output_shape[2] if output_shape[2] else 1
                c = input_shape[3] if input_shape[3] else 1
                # MACs = H * W * C * K * K (no C_out multiplier)
                total += h * w * c * kernel_size * kernel_size
        elif op_name == "UPSAMPLE_2D":
            # Upsample has no MACs, just interpolation.
            # Computation is 0, but reserved for potential future weighting.
            total += 0

        elif op_name == "CONCAT":
            # Concat has no MACs; it's just memory concatenation.
            total += 0

        elif op_name == "ADD":
            # Add is element-wise and typically doesn't count as MACs
            # (though some implementations count 1 op/element).
            # We conservatively estimate it as 0 here.
            total += 0

        elif op_name == "DETECTION_HEAD":
            # Detection heads usually have 1-2 conv layers.
            # We simplify this by reading from params if available.
            # Otherwise, we estimate it as 0 for generality,
            # as the true MACs are covered by internal convolutions.
            total += 0

        elif op_name in ["CONV_TRANSPOSE", "TRANSPOSED_CONV"]:
            # Transposed convolution: Same calculation as standard convolution,
            # but reversed, enlarging H/W.
            conv_params = op.get("conv_params", {})
            kernel_size = conv_params.get("kernel_size", 3)
            stride = conv_params.get("stride", 2)

            input_idx = op.get("input_indices", [0])[0]
            output_idx = op.get("output_indices", [0])[0]

            input_shape = tensors.get(input_idx, {}).get("shape", [1, 1, 1, 1])
            output_shape = tensors.get(output_idx, {}).get("shape",
                                                           [1, 1, 1, 1])

            if len(input_shape) >= 4 and len(output_shape) >= 4:
                h = output_shape[1] if output_shape[1] else 1
                w = output_shape[2] if output_shape[2] else 1
                c_in = input_shape[3] if input_shape[3] else 1
                c_out = output_shape[3] if output_shape[3] else 1
                # MACs of transposed convolution: H * W * C_in * C_out * K * K
                total += h * w * c_in * c_out * kernel_size * kernel_size

    return total


def calculate_params(model_info: Dict[str, Any]) -> int:
    """
    Calculate the total number of parameters.

    Args:
        model_info: ModelInfo dictionary.

    Returns:
        Total parameter count.
    """
    total = 0
    weights = model_info.get("weights", {})

    for weight in weights.values():
        # Weight is a numpy array or a list
        if hasattr(weight, "size"):
            total += weight.size
        elif isinstance(weight, list):
            total += len(weight)

    return total


def calculate_peak_ram(model_info: Dict[str, Any]) -> int:
    """
    Calculate the peak RAM usage of the model.

    This is the maximum size of all tensors that need to be held in
    memory simultaneously at any point during inference.

    Args:
        model_info: ModelInfo dictionary.

    Returns:
        Peak RAM usage in bytes.
    """
    ops = model_info.get("ops", [])
    tensors = model_info.get("tensors", {})

    # Track which tensors are live at each op
    tensor_sizes = {}
    for idx, spec in tensors.items():
        shape = spec.get("shape", [])
        size = 1
        for dim in shape:
            size *= dim
        # Assume int8 = 1 byte per element
        tensor_sizes[idx] = size

    # Simulate execution to find peak memory usage
    live_tensors = set()
    peak = 0

    for op in ops:
        # Input tensors become live before the op
        for idx in op.get("input_indices", []):
            live_tensors.add(idx)

        # Output tensors become live after the op
        for idx in op.get("output_indices", []):
            live_tensors.add(idx)

        # Calculate current memory usage
        current = 0
        for idx in live_tensors:
            current += tensor_sizes.get(idx, 0)

        peak = max(peak, current)

        # Tensors that are no longer needed can be freed
        # For simplicity, we assume only the last output is kept
        # More precise analysis would require liveness tracking

    return peak


def flatten_weights(weights: Dict[int, np.ndarray]) -> Dict[int, List[int]]:
    """
    Flatten all weight tensors into lists.

    Args:
        weights: Dictionary of weight tensors (numpy arrays).

    Returns:
        Dictionary of flattened weight lists.
    """
    result = {}
    for idx, weight in weights.items():
        if hasattr(weight, "flatten"):
            result[idx] = weight.flatten().tolist()
        elif isinstance(weight, list):
            result[idx] = weight
        else:
            result[idx] = [weight]
    return result


def generate_random_weights_from_structure(
    structure: Dict[str, Any],
    seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Generate random weights for a network structure.

    Args:
        structure: Structure dict with 'layers', 'input_shape', 'output_shape'
        seed: Random seed for reproducibility

    Returns:
        Dict mapping tensor index to weight array
    """
    if seed is not None:
        np.random.seed(seed)

    weights = {}
    layers = structure.get("layers", [])
    input_shape = structure.get("input_shape", [1, 28, 28, 1])

    next_idx = 1
    current_shape = list(input_shape)
    current_channels = input_shape[-1] if len(input_shape) >= 2 else 1

    for layer in layers:
        layer_type = layer.get("type")

        if layer_type == "conv":
            kernel = layer.get("kernel", 3)
            channels_out = layer.get("channels", 16)

            # Weight: [K, K, C_in, C_out]
            weight_shape = [kernel, kernel, current_channels, channels_out]
            weight = np.random.uniform(-0.5, 0.5, size=weight_shape)
            weight = np.clip(np.round(weight * 256), -128, 127).astype(np.int8)
            weights[next_idx] = weight

            # Bias: [C_out]
            bias_shape = [channels_out]
            bias = np.random.uniform(-0.5, 0.5, size=bias_shape)
            bias = np.clip(np.round(bias * 256), -128, 127).astype(np.int32)
            weights[next_idx + 1] = bias

            next_idx += 2
            current_channels = channels_out

        elif layer_type == "fc":
            units = layer.get("units", 64)

            # Flatten current shape to get input size
            input_size = 1
            for d in current_shape:
                input_size *= d

            # Weight: [input_size, units]
            weight_shape = [input_size, units]
            weight = np.random.uniform(-0.5, 0.5, size=weight_shape)
            weight = np.clip(np.round(weight * 256), -128, 127).astype(np.int8)
            weights[next_idx] = weight

            # Bias: [units]
            bias_shape = [units]
            bias = np.random.uniform(-0.5, 0.5, size=bias_shape)
            bias = np.clip(np.round(bias * 256), -128, 127).astype(np.int32)
            weights[next_idx + 1] = bias

            next_idx += 2

        elif layer_type == "pool":
            # Pool layers have no weights
            pass

        elif layer_type == "detection_head":
            # Detection head is a placeholder, no weights for now
            pass

    return weights


def _random_int8_weight(shape: List[int]) -> np.ndarray:
    """
    Generate random int8 weight tensor.

    Values are uniformly distributed in [-128, 127] range.

    Args:
        shape: Shape of the weight tensor.

    Returns:
        int8 numpy array.
    """
    # Generate random values in [-128, 127] range
    # Use uniform distribution centered at 0
    weight = np.random.uniform(-0.5, 0.5, size=shape)
    # Scale to int8 range: [-128, 127]
    # 0.5 * 256 = 128, so this maps [-0.5, 0.5] to [-128, 127]
    weight = np.round(weight * 256).astype(np.int8)
    return weight


def _random_int32_bias(shape: List[int]) -> np.ndarray:
    """
    Generate random int32 bias tensor.

    Values are uniformly distributed in [-128, 127] range.

    Args:
        shape: Shape of the bias tensor.

    Returns:
        int32 numpy array.
    """
    # Use same range as weights for consistency
    bias = np.random.uniform(-0.5, 0.5, size=shape)
    bias = np.round(bias * 256).astype(np.int32)
    return bias


def fill_model_info_with_weights(
    model_info: Dict[str, Any],
    weights: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    """
    Fill a model_info dictionary with weight data.
    """
    result = copy.deepcopy(model_info)
    result["weights"] = {}
    for idx, weight in weights.items():
        if isinstance(weight, np.ndarray):
            result["weights"][str(idx)] = weight.tolist()
        else:
            result["weights"][str(idx)] = weight
    return result
