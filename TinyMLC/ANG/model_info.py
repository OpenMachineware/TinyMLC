# model_info.py
# Unified Intermediate Representation (IR) for TinyMLC.
# This structure is used as the contract between all frontends
# (LiteRT, ONNX, ANG) and the backend code generator.

from typing import Dict, List, Optional, Any, Union
import numpy as np


class TensorSpec:
    """
    Specification of a tensor in the model.

    Attributes:
        name: Unique identifier for the tensor.
        shape: List of dimensions (batch, height, width, channels, ...).
        dtype: Data type of tensor values (e.g., "int8", "int32", "float32").
        scale: Quantization scale factor (optional, None for float models).
        zero_point: Quantization zero point (optional, None for float models).
    """

    def __init__(
        self,
        name: str,
        shape: List[int],
        dtype: str,
        tensor_index: Optional[int] = None,
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.tensor_index = tensor_index
        self.scale = scale
        self.zero_point = zero_point

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "tensor_index": self.tensor_index,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }


class Op:
    """
    A single operation/layer in the model.

    Attributes:
        index: Unique index for this operation.
        op_name: Type of operation (e.g., "CONV_2D", "FULLY_CONNECTED").
        input_indices: List of tensor indices that are inputs to this op.
        output_indices: List of tensor indices that are outputs of this op.
        params: Op-specific parameters (conv_params, fc_params, etc.).
        state: Operator state ("created", "translated", "generated").
        pass_flags: Dictionary of pass flags.
    """

    def __init__(
        self,
        op_name: str,
        input_indices: List[int],
        output_indices: List[int],
        params: Optional[Dict[str, Any]] = None,
        index: Optional[int] = None,
        state: str = "created",
    ):
        self.index = index
        self.op_name = op_name
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.params = params or {}
        self.state = state
        self.pass_flags = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        result = {
            "index": self.index,
            "op_name": self.op_name,
            "input_indices": self.input_indices,
            "output_indices": self.output_indices,
            "state": self.state,
            "pass_flags": self.pass_flags,
        }
        result.update(self.params)
        return result


class ModelInfo:
    """
    Unified Intermediate Representation for neural network models.

    This is the central data structure used throughout TinyMLC.
    All frontends (LiteRT, ONNX, ANG) produce this structure.
    The backend code generator consumes this structure.

    Attributes:
        inputs: List of input tensor specifications.
        outputs: List of output tensor specifications.
        ops: List of operations in execution order.
        tensors: Dictionary mapping tensor index to TensorSpec.
        weights: Dictionary mapping tensor index to numpy array data.
        quant_scales: Global quantization scales (if not per-tensor).
    """

    def __init__(
        self,
        inputs: List[TensorSpec],
        outputs: List[TensorSpec],
        ops: List[Op],
        tensors: Dict[int, TensorSpec],
        weights: Dict[int, np.ndarray],
        quant_scales: Optional[Dict[str, Any]] = None,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.ops = ops
        self.tensors = tensors
        self.weights = weights
        self.quant_scales = quant_scales or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire ModelInfo to a dictionary."""
        return {
            "input": [t.to_dict() for t in self.inputs],
            "output": [t.to_dict() for t in self.outputs],
            "ops": [op.to_dict() for op in self.ops],
            "tensors": {
                idx: spec.to_dict() for idx, spec in self.tensors.items()
            },
            "weights": {
                str(idx): weight.tolist()
                for idx, weight in self.weights.items()
            },
            "quant_scales": self.quant_scales,
        }

    def get_tensor(self, index: int) -> Optional[TensorSpec]:
        """Get tensor specification by index."""
        return self.tensors.get(index)

    def get_weight(self, index: int) -> Optional[np.ndarray]:
        """Get weight data by tensor index."""
        return self.weights.get(index)

    def add_tensor(
        self,
        index: int,
        spec: TensorSpec,
        weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a tensor to the model.

        Args:
            index: Unique index for the tensor.
            spec: TensorSpec describing the tensor.
            weight: Optional numpy array with weight data.
        """
        self.tensors[index] = spec
        if weight is not None:
            self.weights[index] = weight

    def add_op(self, op: Op) -> None:
        """Add an operation to the model."""
        self.ops.append(op)

    def validate(self) -> bool:
        """
        Validate the model for consistency.

        Checks:
            - All indices in ops refer to valid tensors.
            - All weights have matching shapes with their tensor specs.
            - Input/output lists are not empty.

        Returns:
            True if the model is valid, False otherwise.
        """
        # Check that input and output are not empty
        if not self.inputs or not self.outputs:
            return False

        # Check that all tensor indices in ops exist
        all_tensor_indices = set(self.tensors.keys())
        for op in self.ops:
            for idx in op.input_indices + op.output_indices:
                if idx not in all_tensor_indices:
                    return False

        # Check that weights match tensor shapes
        for idx, weight in self.weights.items():
            if idx not in self.tensors:
                return False
            expected_shape = self.tensors[idx].shape
            if list(weight.shape) != expected_shape:
                # Allow broadcasting for scalar weights (shape mismatch is OK)
                if len(weight.shape) != 0:
                    return False

        return True


def default_quant_scale() -> float:
    """
    Return the default quantization scale for int8 models.

    This is the scale corresponding to 1/256, which is a common default
    when no specific scale is provided.

    Returns:
        Default scale value.
    """
    # 1/256 = 0.00390625
    return 1.0 / 256.0


def default_zero_point() -> int:
    """
    Return the default zero point for int8 models.

    Returns:
        Default zero point (0 for symmetric quantization).
    """
    return 0


def create_default_tensor_spec(
    name: str,
    shape: List[int],
    dtype: str = "int8",
) -> TensorSpec:
    """
    Create a tensor specification with default quantization parameters.

    Args:
        name: Tensor name.
        shape: Tensor shape.
        dtype: Data type (default: "int8").

    Returns:
        A TensorSpec with default scale and zero point if quantized.
    """
    if dtype.startswith("int"):
        return TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            scale=default_quant_scale(),
            zero_point=default_zero_point(),
        )
    else:
        return TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            scale=None,
            zero_point=None,
        )
