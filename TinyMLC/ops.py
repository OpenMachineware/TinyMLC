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
