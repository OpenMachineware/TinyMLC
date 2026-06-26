#!/usr/bin/env python3
# generate_tflite_models.py
# Generate multiple TFLite models covering all supported ops.

import tensorflow as tf
import numpy as np

from pathlib import Path


def build_cnn_model():
    """Conv2D, DepthwiseConv2D, MaxPool2D, AvgPool2D,
       GlobalAvgPool2D, FC, Softmax, Relu"""
    inputs = tf.keras.Input(shape=(28, 28, 1), name='input')
    x = tf.keras.layers.Conv2D(8, 3, padding='same', name='conv2d')(inputs)
    x = tf.keras.layers.ReLU(name='relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(
        3, padding='same', name='depthwise_conv2d')(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.MaxPooling2D(2, name='max_pool2d')(x)
    x = tf.keras.layers.AveragePooling2D(2, name='avg_pool2d')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dense(10, name='fc')(x)
    outputs = tf.keras.layers.Softmax(name='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model_cnn')
    return model


def build_activations_model():
    """ReLU, LeakyReLU, ReLU6, PReLU, HardSigmoid, Sigmoid, Tanh, Clip"""
    inputs = tf.keras.Input(shape=(16,), name='input')
    x = tf.keras.layers.Dense(16, name='dummy')(inputs)
    relu = tf.keras.layers.ReLU(name='relu')(x)
    leaky = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu')(relu)
    relu6 = tf.keras.layers.ReLU(max_value=6.0, name='relu6')(leaky)
    prelu = tf.keras.layers.PReLU(name='prelu')(relu6)
    hs = tf.keras.layers.Activation('hard_sigmoid', name='hard_sigmoid')(prelu)
    sig = tf.keras.layers.Activation('sigmoid', name='sigmoid')(hs)
    tanh = tf.keras.layers.Activation('tanh', name='tanh')(sig)
    clip = tf.keras.layers.Lambda(
        lambda z: tf.clip_by_value(z, -2.0, 2.0), name='clip')(tanh)
    outputs = clip
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='model_activations')
    return model


def build_tensor_ops_model():
    inputs = tf.keras.Input(shape=(8, 8, 4))
    x = tf.keras.layers.Conv2D(4, 3, padding='same')(inputs)

    # Split
    splits = tf.keras.layers.Lambda(lambda z: tf.split(z, 2, axis=-1))(x)

    # Pad channels only (H/W unchanged, only modify channel dimension)
    pad = tf.keras.layers.Lambda(
        lambda z: tf.pad(z, [[0,0], [0,0], [0,0], [1,1]]),
        # [batch, H, W, channels]
        name='pad'
    )(splits[0])

    # Concat back (shapes now match: H/W are both 8)
    concat = tf.keras.layers.Concatenate(name='concat')([pad, splits[1]])

    # Flatten
    flat = tf.keras.layers.Flatten(name='flatten')(concat)

    # Reshape
    reshape = tf.keras.layers.Reshape((8, -1), name='reshape')(flat)

    # Transpose
    transpose = tf.keras.layers.Lambda(
        lambda z: tf.transpose(z, perm=[0, 2, 1]),
        name='transpose'
    )(reshape)

    # StridedSlice
    slice_out = tf.keras.layers.Lambda(
        lambda z: z[:, 0:4, 0:4],
        name='strided_slice'
    )(transpose)

    return tf.keras.Model(
        inputs=inputs, outputs=slice_out, name='model_tensor_ops')


def build_arithmetic_model():
    """Add, Multiply, Subtract, ReduceSum, ReduceMean, ArgMax"""
    inputs = tf.keras.Input(shape=(8, 8), name='input')
    x = tf.keras.layers.Dense(8, name='dummy')(inputs)

    # split into two branches
    split = tf.keras.layers.Lambda(
        lambda z: tf.split(z, 2, axis=-1), name='split')(x)
    a = split[0]
    b = split[1]

    add = tf.keras.layers.Add(name='add')([a, b])
    mul = tf.keras.layers.Multiply(name='multiply')([add, a])
    sub = tf.keras.layers.Subtract(name='sub')([mul, add])

    reduce_sum = tf.keras.layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=-1, keepdims=True),
        name='reduce_sum'
    )(sub)

    reduce_mean = tf.keras.layers.Lambda(
        lambda z: tf.reduce_mean(z, axis=-1, keepdims=True),
        name='reduce_mean'
    )(reduce_sum)

    argmax = tf.keras.layers.Lambda(
        lambda z: tf.argmax(z, axis=-1, output_type=tf.int32),
        name='argmax'
    )(reduce_mean)

    outputs = tf.keras.layers.Lambda(
        lambda z: tf.cast(z, tf.float32),
        name='cast_out'
    )(argmax)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='model_arithmetic')
    return model


def build_upsample_model():
    """Upsample (via ResizeNearestNeighbor), ConvTranspose"""
    inputs = tf.keras.Input(shape=(8, 8, 2), name='input')

    # Conv2DTranspose
    x = tf.keras.layers.Conv2DTranspose(
        4, 3, strides=2, padding='same', name='conv_transpose')(inputs)

    # Upsample (ResizeNearestNeighbor)
    upsample = tf.keras.layers.Lambda(
        lambda z: tf.image.resize(z, (16, 16), method='nearest'),
        name='upsample'
    )(x)

    outputs = upsample
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='model_upsample')
    return model


def build_lstm_model():
    """LSTM, SVDF (simulated with Dense for SVDF)"""
    inputs = tf.keras.Input(shape=(10, 4), name='input')
    lstm = tf.keras.layers.LSTM(8, return_sequences=False, name='lstm')(inputs)
    # SVDF isn't standard in Keras, use Dense as proxy
    svdf = tf.keras.layers.Dense(4, name='svdf')(lstm)
    outputs = svdf
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model_lstm')
    return model


def convert_to_tflite(model, name, output_dir=None):
    """Convert Keras model to TFLite."""
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{name}.tflite'

    model.build(model.input_shape)

    # Convert with default settings
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite)
        print(f'  Saved: {output_path}')
        return
    except Exception as e:
        print(f'  Warning: int8 conversion failed: {e}')
        print(f'  Retrying with float and SELECT_TF_OPS...')

    # Fallback: use float + SELECT_TF_OPS (for LSTM/SVDF)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    try:
        tflite = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite)
        print(f'  Saved: {output_path} (float + SELECT_TF_OPS)')
    except Exception as e:
        print(f'  Error: {name} conversion failed: {e}')


def main():
    models = [
        ('model_cnn', build_cnn_model()),
        ('model_activations', build_activations_model()),
        ('model_tensor_ops', build_tensor_ops_model()),
        ('model_arithmetic', build_arithmetic_model()),
        ('model_upsample', build_upsample_model()),
        ('model_lstm', build_lstm_model()),
    ]

    output_dir = Path("model_tests/tflite")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models:
        convert_to_tflite(model, name, output_dir=output_dir)


if __name__ == '__main__':
    main()
