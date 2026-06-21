#!/bin/bash
# ARM CMSIS-NN Debug Build Script
#
# This script compiles operators from ./c/ directory where CMSIS-NN
# accelerated versions override pure C implementations.

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
SIM="qemu-system-arm"
ARCH="cortex-m4"
ABI="aapcs"

# Default paths (can be overridden by --acc-lib-inc and --acc-lib-lib)
CMSIS_NN_INC="${ACC_LIB_INC:-../third_party/CMSIS-NN-7.0.0/Include}"
CMSIS_NN_LIB="${ACC_LIB_LIB:-../third_party/CMSIS-NN-7.0.0/Lib/libcmsis-nn.a}"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib \
    -ffreestanding -fno-omit-frame-pointer \
    -DTINYMLC_DEBUG \
    -I./include -I./c -I. -I$CMSIS_NN_INC"
LDFLAGS="-L$(dirname $CMSIS_NN_LIB) \
    -lcmsis-nn -lgcc"

# ========== Compile All Operators (CMSIS-NN overrides pure C) ==========
# These compile from ./c/ where CMSIS-NN versions are copied to
# override pure C implementations
$CC $CFLAGS -c c/fc.c -o fc.o
$CC $CFLAGS -c c/softmax.c -o softmax.o
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/add.c -o add.o
$CC $CFLAGS -c c/multiply.c -o multiply.o
$CC $CFLAGS -c c/sub.c -o sub.o
$CC $CFLAGS -c c/svdf.c -o svdf.o
$CC $CFLAGS -c c/conv2d.c -o conv2d.o
$CC $CFLAGS -c c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o
$CC $CFLAGS -c c/sigmoid.c -o sigmoid.o
$CC $CFLAGS -c c/tanh.c -o tanh.o
$CC $CFLAGS -c c/concat.c -o concat.o

# ========== LSTM (Optional) ==========
if grep -q "HAS_LSTM" model_features.txt 2>/dev/null; then
    $CC $CFLAGS -c lstm/c/lstm.c -o lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="lstm.o lut.o"
else
    LSTM_OBJ=""
fi

# ========== Startup and Debug ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o
$CC $CFLAGS -c main_test.c -o main_test.o

# ========== Link ==========
$CC $CFLAGS -T link_arm.ld \
    start.o debug_print.o \
    fc.o softmax.o conv2d.o reshape.o add.o multiply.o sub.o \
    sigmoid.o tanh.o concat.o svdf.o \
    max_pool2d.o depthwise_conv2d.o relu.o avg_pool2d.o \
    transpose.o pad.o mean.o \
    $LSTM_OBJ \
    model.o main_test.o \
    $LDFLAGS \
    -o model.elf

# ========== Run ==========
$SIM -M mps2-an386 -nographic -semihosting -serial mon:stdio -kernel model.elf
