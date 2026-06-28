# ANG/model_builder.py
# Builder for constructing ModelInfo structures from scratch.

import copy
from typing import List, Optional, Dict
import numpy as np

from TinyMLC.ANG.model_info import ModelInfo, TensorSpec, Op
from TinyMLC.ANG.utils import (calculate_macs, calculate_params,
                               calculate_peak_ram, calculate_flash)


class ModelBuilder:
    """
    Fluent builder for constructing ModelInfo structures.

    This class handles:
        - Sequential layer addition (Conv, Pool, FC, Upsample, Concat)
        - Shape inference and tracking
        - Weight initialization (random or provided)
        - Final ModelInfo assembly
    """

    def __init__(self, model_name: str = "ang_generated"):
        self.model_name = model_name
        self.inputs: List[TensorSpec] = []
        self.outputs: List[TensorSpec] = []
        self.ops: List[Op] = []
        self.tensors: Dict[int, TensorSpec] = {}
        self.weights: Dict[int, np.ndarray] = {}
        self._next_tensor_index: int = 0
        self._layer_counter: int = 0

    # ============ Tensor Management ============

    def _next_tensor(self) -> int:
        """Get the next available tensor index."""
        idx = self._next_tensor_index
        self._next_tensor_index += 1
        return idx

    def _next_layer_name(self, prefix: str) -> str:
        """Generate a unique layer name."""
        self._layer_counter += 1
        return f"{prefix}_{self._layer_counter}"

    def add_tensor(
        self,
        name: str,
        shape: List[int],
        dtype: str = "int8",
        weight: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
    ) -> int:
        """
        Add a tensor to the model.

        Args:
            name: Tensor name.
            shape: Tensor shape.
            dtype: Data type.
            weight: Optional weight data.
            scale: Optional quantization scale.
            zero_point: Optional quantization zero point.

        Returns:
            The tensor index.
        """
        if scale is None and dtype.startswith("int"):
            # Default scale for int8: 1/256
            scale = 1.0 / 256.0
            zero_point = 0

        spec = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            scale=scale,
            zero_point=zero_point,
        )
        idx = self._next_tensor()
        self.tensors[idx] = spec

        if weight is not None:
            self.weights[idx] = weight

        return idx

    def add_input(
        self,
        name: str,
        shape: List[int],
        dtype: str = "int8",
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
    ) -> int:
        """Add an input tensor."""
        if scale is None and dtype.startswith("int"):
            scale = 1.0 / 256.0
            zero_point = 0
        idx = self._next_tensor()

        spec = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            tensor_index=idx,
            scale=scale,
            zero_point=zero_point,
        )
        self.inputs.append(spec)
        self.tensors[idx] = spec
        return idx

    def add_output(
        self,
        name: str,
        shape: List[int],
        dtype: str = "int8",
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
    ) -> int:
        """Add an output tensor."""
        if scale is None and dtype.startswith("int"):
            scale = 1.0 / 256.0
            zero_point = 0
        idx = self._next_tensor()

        spec = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            tensor_index=idx,
            scale=scale,
            zero_point=zero_point,
        )
        self.outputs.append(spec)
        self.tensors[idx] = spec
        return idx

    # ============ Layer Building ============

    def add_conv(
        self,
        input_idx: int,
        output_idx: int,
        kernel_size: int,
        channels_out: int,
        stride: int = 1,
        padding: str = "SAME",
        activation: str = "relu",
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add a Conv2D operation.

        Returns:
            The output tensor index (same as output_idx).
        """
        input_spec = self.tensors.get(input_idx)
        if input_spec is None:
            raise ValueError(f"Input tensor {input_idx} not found")

        if len(input_spec.shape) != 4:
            # If not 4D, treat as 1D conv or raise
            # For now, assume 2D conv on 4D input
            pass

        channels_in = input_spec.shape[3] if len(input_spec.shape) == 4 else 1

        # Create weight tensor
        weight_shape = [kernel_size, kernel_size, channels_in, channels_out]
        weight_name = self._next_layer_name("conv_weight")

        if weight is None:
            weight = self._random_weight(weight_shape)

        weight_idx = self.add_tensor(
            name=weight_name,
            shape=weight_shape,
            dtype="int8",
            weight=weight,
        )

        # Create bias tensor
        bias_shape = [channels_out]
        bias_name = self._next_layer_name("conv_bias")

        if bias is None:
            bias = self._random_bias(bias_shape)

        bias_idx = self.add_tensor(
            name=bias_name,
            shape=bias_shape,
            dtype="int32",
            weight=bias,
        )

        # Get input shape for conv params
        input_shape = input_spec.shape
        output_shape = (
            self.tensors.get(output_idx).shape
            if output_idx in self.tensors
            else input_shape
        )

        # Calculate output shape (simplified)
        out_h = (
            (input_shape[1] + 2 * 0 - kernel_size) // stride + 1
            if len(input_shape) >= 2
            else 1
        )
        out_w = (
            (input_shape[2] + 2 * 0 - kernel_size) // stride + 1
            if len(input_shape) >= 3
            else 1
        )

        # Add Conv2D operation
        op = Op(
            op_name="CONV_2D",
            input_indices=[input_idx, weight_idx, bias_idx],
            output_indices=[output_idx],
            params={
                "data_input_idx": input_idx,
                "conv_params": {
                    "input_h": input_shape[1] if len(input_shape) >= 2 else 1,
                    "input_w": input_shape[2] if len(input_shape) >= 3 else 1,
                    "input_c": (
                        input_shape[3] if len(input_shape) >= 4 else channels_in
                    ),
                    "output_h": out_h,
                    "output_w": out_w,
                    "output_c": channels_out,
                    "kernel_h": kernel_size,
                    "kernel_w": kernel_size,
                    "stride_h": stride,
                    "stride_w": stride,
                    "padding_h": 0,
                    "padding_w": 0,
                }
            },
        )
        self.ops.append(op)

        return output_idx

    def add_fc(
        self,
        input_idx: int,
        output_idx: int,
        units: int,
        activation: str = "relu",
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ) -> int:
        """Add a Fully Connected (Dense) operation."""
        input_spec = self.tensors.get(input_idx)
        if input_spec is None:
            raise ValueError(f"Input tensor {input_idx} not found")

        input_size = 1
        for dim in input_spec.shape:
            input_size *= dim

        # Create weight tensor
        weight_shape = [input_size, units]
        weight_name = self._next_layer_name("fc_weight")

        if weight is None:
            weight = self._random_weight(weight_shape)

        weight_idx = self.add_tensor(
            name=weight_name,
            shape=weight_shape,
            dtype="int8",
            weight=weight,
        )

        # Create bias tensor
        bias_shape = [units]
        bias_name = self._next_layer_name("fc_bias")

        if bias is None:
            bias = self._random_bias(bias_shape)

        bias_idx = self.add_tensor(
            name=bias_name,
            shape=bias_shape,
            dtype="int32",
            weight=bias,
        )

        op = Op(
            op_name="FULLY_CONNECTED",
            input_indices=[input_idx, weight_idx, bias_idx],
            output_indices=[output_idx],
            params={
                "data_input_idx": input_idx,
                "fc_params": {
                    "units": units,
                    "activation": activation,
                }
            },
        )
        self.ops.append(op)

        return output_idx

    # ============ P2 New Ops (Detection / Segmentation) ============

    def add_upsample(
        self,
        input_idx: int,
        output_idx: int,
        scale_factor: int = 2,
        mode: str = "nearest",
    ) -> int:
        """
        Add an Upsample operation.

        Args:
            input_idx: Input tensor index.
            output_idx: Output tensor index.
            scale_factor: Upsampling factor (2, 4, 8).
            mode: Interpolation mode ("nearest" or "bilinear").

        Returns:
            The output tensor index (same as output_idx).
        """
        op = Op(
            op_name="UPSAMPLE_2D",
            input_indices=[input_idx],
            output_indices=[output_idx],
            params={
                "upsample_params": {
                    "scale_factor": scale_factor,
                    "mode": mode,
                }
            },
        )
        self.ops.append(op)
        return output_idx

    def add_concat(
        self,
        input_indices: List[int],
        output_idx: int,
        axis: int = -1,
    ) -> int:
        """
        Add a Concat operation.

        Args:
            input_indices: List of input tensor indices.
            output_idx: Output tensor index.
            axis: Concatenation axis.

        Returns:
            The output tensor index (same as output_idx).
        """
        op = Op(
            op_name="CONCAT",
            input_indices=input_indices,
            output_indices=[output_idx],
            params={
                "concat_params": {
                    "axis": axis,
                }
            },
        )
        self.ops.append(op)
        return output_idx

    def add_add(
        self,
        input_indices: List[int],
        output_idx: int,
    ) -> int:
        """
        Add an Add operation (element-wise addition).

        Args:
            input_indices: List of input tensor indices (usually 2).
            output_idx: Output tensor index.

        Returns:
            The output tensor index (same as output_idx).
        """
        op = Op(
            op_name="ADD",
            input_indices=input_indices,
            output_indices=[output_idx],
            params={},
        )
        self.ops.append(op)
        return output_idx

    def add_detection_head(
        self,
        input_idx: int,
        output_boxes_idx: int,
        output_classes_idx: int,
        num_anchors: int = 3,
        num_classes: int = 10,
    ) -> None:
        """
        Add a Detection Head.

        This is a multi-output head used for object detection.
        It produces:
            - Box predictions: [N, num_anchors, 4]
            - Class predictions: [N, num_anchors, num_classes]

        Args:
            input_idx: Input tensor index (feature map).
            output_boxes_idx: Output tensor index for boxes.
            output_classes_idx: Output tensor index for classes.
            num_anchors: Number of anchors per location.
            num_classes: Number of object classes.
        """
        # Detection head typically is a set of conv layers
        # For now, we add a placeholder op that captures the head
        op = Op(
            op_name="DETECTION_HEAD",
            input_indices=[input_idx],
            output_indices=[output_boxes_idx, output_classes_idx],
            params={
                "detection_params": {
                    "num_anchors": num_anchors,
                    "num_classes": num_classes,
                }
            },
        )
        self.ops.append(op)

    # ============ Weight Initialization ============

    def _random_weight(self, shape: List[int]) -> np.ndarray:
        """Generate random int8 weight."""
        # Uniform distribution in [-0.5, 0.5] scaled to int8
        weight = np.random.uniform(-0.5, 0.5, size=shape)
        weight = np.clip(np.round(weight * 256), -128, 127).astype(np.int8)
        return weight

    def _random_bias(self, shape: List[int]) -> np.ndarray:
        """Generate random int32 bias."""
        bias = np.random.uniform(-0.5, 0.5, size=shape)
        bias = np.clip(np.round(bias * 256), -128, 127).astype(np.int32)
        return bias

    # ============ Build ============

    def build(self) -> ModelInfo:
        """Build and validate the final ModelInfo."""
        # Add index and set state to "generated" for all ops
        for i, op in enumerate(self.ops):
            op.index = i
            op.state = "generated"

        model_info = ModelInfo(
            inputs=self.inputs,
            outputs=self.outputs,
            ops=self.ops,
            tensors=self.tensors,
            weights=self.weights,
            quant_scales={},
        )

        # calc stat info.
        model_dict = model_info.to_dict()
        macs = calculate_macs(model_dict)
        params = calculate_params(model_dict)
        peak_ram = calculate_peak_ram(model_dict)
        flash = calculate_flash(model_dict)

        # Fill quant_scales
        model_info.quant_scales = {
            "macs": macs,
            "params": params,
            "peak_ram": peak_ram,
            "flash": flash,
        }

        if not model_info.validate():
            # Add more detailed validation
            raise ValueError("Model validation failed")

        return model_info
