# 测试模型生成工具
# 
# 用途：为 TinyMLC 生成测试用的 FP32 和 INT8 量化 ONNX 模型
# 生成的网络结构：Conv2d(1, 8, 3x3) -> ReLU -> MaxPool2d(2x2) -> Linear(8*14*14, 10)
# 输入：[1, 1, 28, 28] （MNIST 格式）
# 输出：[1, 10] （10分类）
# 
# 使用方法：
#   cd /path/to/TinyMLC
#   python utils/generate_test_models.py
# 
# 输出文件：
#   - model_fp32.onnx: 原始 FP32 模型
#   - model_int8.onnx: 量化后的 INT8 QDQ 格式模型
# 
# 依赖：
#   - torch >= 2.0
#   - onnx
#   - onnxruntime
# 
# 安装依赖：
#   pip install torch onnx onnxruntime

import torch
import torch.nn as nn
import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader


class SimpleNet(nn.Module):
    """简单的 CNN 网络，用于测试 TinyMLC 的 ONNX 解析和量化功能"""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CalibrationData(CalibrationDataReader):
    """量化校准数据读取器"""
    
    def __init__(self):
        self.enumerator = None
    
    def get_next(self):
        if self.enumerator is None:
            self.enumerator = iter([
                {"input": np.random.randn(1, 1, 28, 28).astype(np.float32)} 
                for _ in range(100)
            ])
        return next(self.enumerator, None)


def main():
    model = SimpleNet()
    model.eval()
    
    print("Exporting FP32 ONNX model...")
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        "test_models/model_fp32.onnx",
        opset_version=10,
        input_names=["input"],
        output_names=["output"],
        verbose=False
    )
    
    print("Quantizing to INT8 QDQ format...")
    quantize_static(
        "test_models/model_fp32.onnx",
        "test_models/model_int8.onnx",
        CalibrationData(),
        weight_type=QuantType.QInt8,
        quant_format=0
    )
    
    print("Done!")
    print("  - test_models/model_fp32.onnx: FP32 model")
    print("  - test_models/model_int8.onnx: INT8 QDQ model")


if __name__ == "__main__":
    main()