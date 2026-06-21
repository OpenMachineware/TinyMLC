# 测试模型生成工具
# 
# 用途：为 TinyMLC 生成测试用的 ONNX 模型，覆盖常用算子
# 
# 使用方法：
#   cd /path/to/TinyMLC
#   python utils/generate_test_models.py
# 
# 输出文件：
#   - model_fp32.onnx: CNN基础模型 (Conv2d+ReLU+MaxPool2d+Linear)
#   - model_int8.onnx: INT8 QDQ格式模型
#   - model_cnn_bn.onnx: CNN+BatchNorm模型
#   - model_cnn_gap.onnx: CNN+GlobalAveragePool模型
#   - model_mlp.onnx: MLP多层感知机模型
#   - model_concat.onnx: Concat拼接模型
#   - model_depthwise.onnx: Depthwise卷积模型
#   - model_sigmoid.onnx: Sigmoid激活模型
#   - model_mul.onnx: Mul乘法模型
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
import os

OUTPUT_DIR = "test_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SimpleNet(nn.Module):
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


class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNWithGAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ConcatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        return x


class DepthwiseConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, groups=1)
        self.pointwise = nn.Conv2d(8, 16, 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 30 * 30, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SigmoidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class MulNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = x1 * x2
        return x


class CalibrationData(CalibrationDataReader):
    def __init__(self):
        self.enumerator = None
    
    def get_next(self):
        if self.enumerator is None:
            self.enumerator = iter([
                {"input": np.random.randn(1, 1, 28, 28).astype(np.float32)} 
                for _ in range(100)
            ])
        return next(self.enumerator, None)


def export_model(model, name, input_shape, input_name="input", opset_version=10):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=[input_name],
        output_names=["output"],
        verbose=False
    )
    
    print(f"  {name}.onnx - input: {input_shape}")


def main():
    print("Exporting FP32 ONNX models...")
    
    export_model(SimpleNet(), "model_fp32", (1, 1, 28, 28))
    export_model(CNNWithBN(), "model_cnn_bn", (1, 1, 32, 32))
    export_model(CNNWithGAP(), "model_cnn_gap", (1, 1, 28, 28))
    export_model(MLP(), "model_mlp", (1, 784))
    export_model(ConcatNet(), "model_concat", (1, 10))
    export_model(DepthwiseConvNet(), "model_depthwise", (1, 1, 32, 32))
    export_model(SigmoidNet(), "model_sigmoid", (1, 10))
    export_model(MulNet(), "model_mul", (1, 10))
    
    print("\nQuantizing model_fp32 to INT8 QDQ format...")
    quantize_static(
        os.path.join(OUTPUT_DIR, "model_fp32.onnx"),
        os.path.join(OUTPUT_DIR, "model_int8.onnx"),
        CalibrationData(),
        weight_type=QuantType.QInt8,
        quant_format=0
    )
    
    print("\nDone! All models exported to test_models/")


if __name__ == "__main__":
    main()
