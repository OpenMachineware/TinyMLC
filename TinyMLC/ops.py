# -*- coding: utf-8 -*-
# TinyMLC - Tiny Machine Learning Compiler
#
# Copyright (c) 2026 Jia Liu & TinyMLC Contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of TinyMLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


SUPPORTED_OPS = [
    # Activation
    "RELU",
    "RELU6",
    "LEAKY_RELU",
    "PRELU",
    "HARD_SIGMOID",
    "SIGMOID",
    "TANH",
    "CLIP",

    # Convolution
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_TRANSPOSE",

    # Pooling
    "MAX_POOL_2D",
    "AVG_POOL_2D",
    "GLOBAL_AVG_POOL",

    # Fully Connected
    "FULLY_CONNECTED",

    # Activation (continued)
    "SOFTMAX",

    # Tensor operations
    "RESHAPE",
    "TRANSPOSE",
    "CONCAT",
    "SPLIT",
    "PAD",
    "STRIDED_SLICE",
    "FLATTEN",

    # Arithmetic
    "ADD",
    "MULTIPLY",
    "SUB",
    "MEAN",
    "REDUCE_SUM",
    "ARGMAX",

    # Upsampling
    "UPSAMPLE",
    "RESIZE_NEAREST_NEIGHBOR",

    # RNN
    "LSTM",
    "SVDF",

    # Quantization
    "QUANTIZE",
    "DEQUANTIZE",
]
