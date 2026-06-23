#!/usr/bin/env python3
"""Test script to verify all operators generate correct C code"""

import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinymlc.ang.model_info import TensorSpec, Op, ModelInfo
import numpy as np


def create_model_info(ops_data):
    """Create a simple ModelInfo for testing

    Args:
        ops_data: list of dicts with keys:
            - op_name: operator type
            - input_shapes: list of input tensor shapes
            - output_shapes: list of output tensor shapes
            - params: operator parameters (optional)
    """
    tensors = {}
    ops = []
    tensor_idx = 0
    op_idx = 0

    for op_data in ops_data:
        input_indices = []
        for shape in op_data.get("input_shapes", []):
            tensors[tensor_idx] = {
                "name": f"tensor_{tensor_idx}",
                "shape": shape,
                "dtype": "int8",
                "size": int(np.prod(shape)),
                "scale": 0.1,
                "zero_point": 0,
            }
            input_indices.append(tensor_idx)
            tensor_idx += 1

        output_indices = []
        for shape in op_data.get("output_shapes", []):
            tensors[tensor_idx] = {
                "name": f"tensor_{tensor_idx}",
                "shape": shape,
                "dtype": "int8",
                "size": int(np.prod(shape)),
                "scale": 0.1,
                "zero_point": 0,
            }
            output_indices.append(tensor_idx)
            tensor_idx += 1

        op = {
            "index": op_idx,
            "op_name": op_data["op_name"],
            "inputs": input_indices,
            "outputs": output_indices,
            "input_indices": input_indices,
            "output_indices": output_indices,
        }
        op.update(op_data.get("params", {}))
        ops.append(op)
        op_idx += 1

    # Input tensor
    first_op = ops_data[0]
    input_shape = first_op["input_shapes"][0]
    input_tensor = {
        "name": "input",
        "shape": input_shape,
        "dtype": "int8",
        "scale": 0.1,
        "zero_point": 0,
    }

    # Output tensor
    last_op = ops_data[-1]
    output_shape = last_op["output_shapes"][0]
    output_tensor = {
        "name": "output",
        "shape": output_shape,
        "dtype": "int8",
        "scale": 0.1,
        "zero_point": 0,
    }

    model_info = {
        "input": [input_tensor],
        "output": [output_tensor],
        "ops": ops,
        "weights": {},
        "quant_scales": {},
        "tensors": tensors,
    }

    return model_info


def test_operator(op_name, model_info, target="riscv", accel="pure-c"):
    """Test if an operator can be converted to C code"""
    from tinymlc.codegen import generate_c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = generate_c_code(
                model_info=model_info,
                output_dir=tmpdir,
                target=target,
                with_test_main=True,
            )

            # Check if model.c was generated
            if "model.c" in result:
                return True, "OK"
            else:
                return False, "No model.c generated"

        except Exception as e:
            return False, str(e)


def main():
    print("Testing C operators...")
    print()

    # Test configurations for each operator
    test_cases = [
        {"name": "SOFTMAX", "op_name": "SOFTMAX", "input_shapes": [[1, 10]], "output_shapes": [[1, 10]]},
        {"name": "LEAKY_RELU", "op_name": "LEAKY_RELU", "input_shapes": [[1, 32]], "output_shapes": [[1, 32]], "params": {"alpha_q7": 25}},
        {"name": "RELU6", "op_name": "RELU6", "input_shapes": [[1, 32]], "output_shapes": [[1, 32]]},
        {"name": "HARD_SIGMOID", "op_name": "HARD_SIGMOID", "input_shapes": [[1, 32]], "output_shapes": [[1, 32]]},
        {"name": "PRELU", "op_name": "PRELU", "input_shapes": [[1, 4, 4, 16], [16]], "output_shapes": [[1, 4, 4, 16]], "params": {"alpha": np.random.randint(-50, 50, 16).tolist()}},
        {"name": "CLIP", "op_name": "CLIP", "input_shapes": [[1, 32]], "output_shapes": [[1, 32]], "params": {"min_val": -10, "max_val": 10}},
        {"name": "REDUCE_SUM", "op_name": "REDUCE_SUM", "input_shapes": [[1, 4, 4, 16]], "output_shapes": [[1, 4, 4]], "params": {"axis": 3}},
        {"name": "ARGMAX", "op_name": "ARGMAX", "input_shapes": [[1, 4, 4, 16]], "output_shapes": [[1, 4, 4]], "params": {"axis": 3}},
        {"name": "FLATTEN", "op_name": "FLATTEN", "input_shapes": [[1, 4, 4, 16]], "output_shapes": [[1, 256]]},
        {"name": "SPLIT", "op_name": "SPLIT", "input_shapes": [[1, 32]], "output_shapes": [[1, 16], [1, 16]], "params": {"axis": 1, "split_sizes": [16, 16]}},
        {"name": "STRIDED_SLICE", "op_name": "STRIDED_SLICE", "input_shapes": [[1, 8, 8, 16]], "output_shapes": [[1, 4, 4, 8]], "params": {"begin": [0, 2, 2, 0], "end": [1, 6, 6, 8], "strides": [1, 1, 1, 1]}},
        {"name": "UPSAMPLE", "op_name": "UPSAMPLE", "input_shapes": [[1, 4, 4, 16]], "output_shapes": [[1, 8, 8, 16]], "params": {"scale_h": 2, "scale_w": 2}},
        {"name": "TRANSPOSE", "op_name": "TRANSPOSE", "input_shapes": [[1, 4, 8, 16]], "output_shapes": [[1, 8, 4, 16]], "params": {"perm": [0, 2, 1, 3]}},
        {"name": "PAD", "op_name": "PAD", "input_shapes": [[1, 4, 4, 16]], "output_shapes": [[1, 6, 6, 16]], "params": {"paddings": [0, 0, 1, 1, 1, 1, 0, 0]}},
        {"name": "MEAN", "op_name": "MEAN", "input_shapes": [[1, 4, 4, 16]], "output_shapes": [[1, 1, 1, 16]], "params": {"axis": [1, 2], "keep_dims": True}},
        {"name": "CONV_TRANSPOSE", "op_name": "CONV_TRANSPOSE", "input_shapes": [[1, 4, 4, 8]], "output_shapes": [[1, 8, 8, 16]], "params": {"kernel_h": 3, "kernel_w": 3, "stride_h": 2, "stride_w": 2, "pad_h": 1, "pad_w": 1, "out_channels": 16}},
    ]

    results = []
    for tc in test_cases:
        print(f"Testing {tc['name']}...", end=" ")
        model_info = create_model_info([tc])
        success, msg = test_operator(tc["name"], model_info)
        results.append((tc["name"], success, msg))
        print("OK" if success else f"FAILED: {msg}")

    print()
    print("=" * 50)
    print("Summary:")
    passed = sum(1 for _, s, _ in results if s)
    failed = sum(1 for _, s, _ in results if not s)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()
    if failed > 0:
        print("Failed operators:")
        for name, _, msg in results:
            if not _:
                print(f"  - {name}: {msg}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
