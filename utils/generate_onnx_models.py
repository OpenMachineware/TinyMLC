#!/usr/bin/env python3
# generate_onnx_models.py
# Generate multiple ONNX models covering all supported ops.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort

from pathlib import Path


class CNNModel(nn.Module):
    """Conv2D, DepthwiseConv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D, FC, Softmax, ReLU"""
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = self.global_avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class ActivationsModel(nn.Module):
    """ReLU, LeakyReLU, ReLU6, PReLU, Sigmoid, Tanh, Clip"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.1)
        self.prelu = nn.PReLU(16)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.leaky(x)
        x = self.relu6(x)  # ReLU6
        x = self.prelu(x)
        x = self.sigmoid(x)
        x = self.tanh(x)
        x = torch.clamp(x, -2.0, 2.0)  # Clip
        return x

    def relu6(self, x):
        return torch.clamp(x, 0, 6)


class TensorOpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)                     # [1, 4, 8, 8]
        split1 = x[:, :, :, :4]              # [1, 4, 8, 4]
        split2 = x[:, :, :, 4:]              # [1, 4, 8, 4]
        # 两个张量形状完全一样，dim=3 对齐
        concat = torch.cat([split1, split2], dim=3)  # [1, 4, 8, 8]
        # 不需要 pad 了，直接走后面的操作
        flat = concat.view(concat.size(0), -1)
        reshape = flat.view(flat.size(0), 4, -1)
        transpose = reshape.transpose(1, 2)
        return transpose


class ArithmeticModel(nn.Module):
    """Add, Multiply, Subtract, ReduceSum, ReduceMean"""
    def forward(self, x):
        a = x[:, :4]
        b = x[:, 4:8]
        add = a + b
        mul = add * a
        sub = mul - add
        reduce_sum = sub.sum(dim=1, keepdim=True)
        reduce_mean = reduce_sum.mean(dim=1, keepdim=True)
        return reduce_mean


class UpsampleModel(nn.Module):
    """Upsample (nearest), ConvTranspose"""
    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x


class LSTMModel(nn.Module):
    """LSTM, SVDF (simulated)"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 8, batch_first=True)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x


def export_to_onnx(model, name, input_shape, output_dir=None, dynamic_axes=None):
    """Export PyTorch model to ONNX."""
    if output_dir is None:
        output_dir = Path(".")
    output_path = output_dir / f'{name}.onnx'
    model.eval()
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )
    print(f'  Saved: {output_path}')


def main():
    models = [
        ('model_cnn', CNNModel(), (1, 1, 28, 28)),
        ('model_activations', ActivationsModel(), (1, 16)),
        ('model_tensor_ops', TensorOpsModel(), (1, 1, 8, 8)),
        ('model_arithmetic', ArithmeticModel(), (1, 8)),
        ('model_upsample', UpsampleModel(), (1, 4, 4, 4)),
        ('model_lstm', LSTMModel(), (1, 10, 4)),
    ]

    output_dir = Path("model_tests/onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model, input_shape in models:
        try:
            export_to_onnx(model, name, input_shape, output_dir=output_dir)
        except Exception as e:
            print(f'  Warning: {name} export failed: {e}')


if __name__ == '__main__':
    main()
