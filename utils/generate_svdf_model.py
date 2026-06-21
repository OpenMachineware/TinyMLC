# SVDF 测试模型生成工具
# 
# 用途：为 TinyMLC 生成测试用的 SVDF ONNX 模型
# 
# SVDF (Singular Value Decomposition) 是一种常用于语音识别的算子，
# 通过对权重矩阵进行奇异值分解来减少参数数量。
# 
# 使用方法：
#   cd /path/to/TinyMLC
#   python utils/generate_svdf_model.py
# 
# 输出文件：
#   - test_models/model_svdf.onnx: SVDF 测试模型
# 
# 网络结构：
#   输入 [batch, time_steps, input_size] -> SVDF -> 输出 [batch, time_steps, rank*units]
# 
# 依赖：
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

    X = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output_shape)

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