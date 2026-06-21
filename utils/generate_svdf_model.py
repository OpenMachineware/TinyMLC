# SVDF Test Model Generation Tool
#
# Purpose: Generate SVDF ONNX test model for TinyMLC
#
# SVDF (Singular Value Decomposition) is an operator commonly used in
# speech recognition, which reduces parameter count through singular
# value decomposition of weight matrix.
#
# Usage:
#   cd /path/to/TinyMLC
#   python utils/generate_svdf_model.py
#
# Output files:
#   - test_models/model_svdf.onnx: SVDF test model
#
# Network structure:
#   Input [batch, time_steps, input_size] -> SVDF -> Output
#   [batch, time_steps, rank*units]
#
# Dependencies:
#   - onnx
#   - numpy

import onnx
from onnx import helper, numpy_helper
import numpy as np


def generate_svdf_model():
    batch_size = 1
    time_steps = 10
    input_size = 16
    rank = 2
    units = 8
    output_size = rank * units

    input_shape = [batch_size, time_steps, input_size]
    output_shape = [batch_size, time_steps, output_size]
    
    weights_shape = [rank * units, input_size]
    bias_shape = [rank * units]

    np.random.seed(42)
    weights = np.random.randn(*weights_shape).astype(np.float32) * 0.1
    bias = np.random.randn(*bias_shape).astype(np.float32) * 0.1

    X = helper.make_tensor_value_info(
        'input', onnx.TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, output_shape)

    W = numpy_helper.from_array(weights, 'weights')
    B = numpy_helper.from_array(bias, 'bias')

    svdf_node = helper.make_node(
        'SVDF',
        inputs=['input', 'weights', 'bias'],
        outputs=['output'],
        rank=rank,
        activation_function='Tanh',
    )

    graph = helper.make_graph(
        [svdf_node],
        'SVDF_Test',
        [X],
        [Y],
        initializer=[W, B],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])

    output_path = 'test_models/model_svdf.onnx'
    onnx.save(model, output_path)
    print(f"SVDF model saved to: {output_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output_shape}")
    print(f"  Rank: {rank}, Units: {units}")
    print(f"  Weights shape: {weights_shape}")


if __name__ == "__main__":
    generate_svdf_model()