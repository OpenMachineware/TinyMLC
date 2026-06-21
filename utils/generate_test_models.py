# Test Model Generation Tool
#
# Purpose: Generate ONNX test models for TinyMLC, covering common operators
#
# Usage:
#   cd /path/to/TinyMLC
#   python utils/generate_test_models.py
#
# Output files:
#   - model_fp32.onnx: Basic CNN model (Conv2d+ReLU+MaxPool2d+Linear)
#   - model_int8.onnx: INT8 QDQ format model
#   - model_cnn_bn.onnx: CNN+BatchNorm model
#   - model_cnn_gap.onnx: CNN+GlobalAveragePool model
#   - model_mlp.onnx: MLP model
#   - model_mlp_deep.onnx: Deep MLP (5 FC layers)
#   - model_concat.onnx: Concat model
#   - model_depthwise.onnx: Depthwise conv model
#   - model_sigmoid.onnx: Sigmoid activation model
#   - model_mul.onnx: Mul model
#   - model_tanh.onnx: Tanh activation model
#   - model_sub.onnx: Sub model
#   - model_residual.onnx: Residual connection model
#   - model_lstm.onnx: LSTM network model
#
# Dependencies:
#   - torch >= 2.0
#   - onnx
#   - onnxruntime
#
# Install dependencies:
#   pip install torch onnx onnxruntime

import torch
import torch.nn as nn
import onnx
import numpy as np
from onnxruntime.quantization import (
    quantize_static, QuantType, CalibrationDataReader)
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
    """3-layer MLP"""
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


class DeepMLP(nn.Module):
    """5-layer deep MLP"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
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


class TanhNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class SubNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = x1 - x2
        return x


class ResidualNet(nn.Module):
    """Residual connection network"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + identity  # residual connection
        return x


class LSTMNet(nn.Module):
    """Simple LSTM network"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=8, hidden_size=16,
            num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, (h_n, c_n) = self.lstm(x)
        # take output of the last time step
        out = out[:, -1, :]
        x = self.fc(out)
        return x


class GlobalMaxPoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.relu = nn.ReLU()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.gmp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


def export_model(model, name, input_shape,
                 input_name="input", opset_version=11):
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

    # Basic models
    export_model(SimpleNet(), "model_fp32", (1, 1, 28, 28))
    export_model(CNNWithBN(), "model_cnn_bn", (1, 1, 32, 32))
    export_model(CNNWithGAP(), "model_cnn_gap", (1, 1, 28, 28))

    # MLP models
    export_model(MLP(), "model_mlp", (1, 784))
    export_model(DeepMLP(), "model_mlp_deep", (1, 784))

    # Operator test models
    export_model(ConcatNet(), "model_concat", (1, 10))
    export_model(DepthwiseConvNet(), "model_depthwise", (1, 1, 32, 32))
    export_model(SigmoidNet(), "model_sigmoid", (1, 10))
    export_model(MulNet(), "model_mul", (1, 10))
    export_model(TanhNet(), "model_tanh", (1, 10))
    export_model(SubNet(), "model_sub", (1, 10))
    export_model(ResidualNet(), "model_residual", (1, 10))
    
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