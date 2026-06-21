#!/bin/bash
# ARM CMSIS-NN Release Build Script
#
# Compile operators from ./c/ directory.
# CMSIS-NN accelerated operators override pure C implementations.

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
ARCH="cortex-m4"
ABI="aapcs"

# Default paths (can be overridden by --acc-lib-inc and --acc-lib-lib)
CMSIS_NN_INC="${ACC_LIB_INC:-../third_party/CMSIS-NN-7.0.0/Include}"
CMSIS_NN_LIB="${ACC_LIB_LIB:-../third_party/CMSIS-NN-7.0.0/Lib/libcmsis-nn.a}"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib \
    -ffreestanding -fno-omit-frame-pointer \
    -I./include -I./c -I. -I$CMSIS_NN_INC"
LDFLAGS="-L$(dirname $CMSIS_NN_LIB) \
    -lcmsis-nn -lgcc"

# ========== CMSIS-NN Accelerated Operators ==========
$CC $CFLAGS -c c/fc.c -o fc.o
$CC $CFLAGS -c c/conv2d.c -o conv2d.o
$CC $CFLAGS -c c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c c/softmax.c -o softmax.o
$CC $CFLAGS -c c/add.c -o add.o
$CC $CFLAGS -c c/multiply.c -o multiply.o

# ========== Pure C Operators (No CMSIS-NN acceleration) ==========
$CC $CFLAGS -c c/sigmoid.c -o sigmoid.o
$CC $CFLAGS -c c/tanh.c -o tanh.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/concat.c -o concat.o
$CC $CFLAGS -c c/sub.c -o sub.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o
$CC $CFLAGS -c c/svdf.c -o svdf.o

# ========== LSTM (Pure C, uses LUT) ==========
if grep -q "HAS_LSTM" model_features.txt 2>/dev/null; then
    $CC $CFLAGS -c lstm/c/lstm.c -o lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="lstm.o lut.o"
else
    LSTM_OBJ=""
fi

# ========== Startup ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o

# ========== Link ==========
$CC $CFLAGS -T link_arm.ld \
    start.o debug_print.o \
    fc.o conv2d.o depthwise_conv2d.o avg_pool2d.o max_pool2d.o \
    softmax.o add.o multiply.o \
    sigmoid.o tanh.o relu.o reshape.o concat.o sub.o \
    transpose.o pad.o mean.o svdf.o \
    $LSTM_OBJ \
    model.o \
    $LDFLAGS \
    -o model.elf

echo "Release build complete: model.elf"
