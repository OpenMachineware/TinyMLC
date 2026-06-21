#!/bin/bash
# ARM CMSIS-NN Release Build Script
# (no main_test.c generation, no run)

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
ARCH="cortex-m4"
ABI="aapcs"
CMSIS_NN_INC="../third_party/CMSIS-NN-7.0.0/Include"
CMSIS_NN_LIB="../third_party/CMSIS-NN-7.0.0/Lib/libcmsis-nn.a"

CMSIS_NN_INC="${CMSIS_NN_INC:-/opt/CMSIS-NN/Include}"
CMSIS_NN_LIB="${CMSIS_NN_LIB:-/opt/CMSIS-NN/Lib/libcmsis-nn.a}"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib \
    -ffreestanding -fno-omit-frame-pointer \
    -I./include -I./c -I. -I$CMSIS_NN_INC"

# ========== Compile ARM Accelerated Operators ==========
$CC $CFLAGS -c fc.c -o fc.o
$CC $CFLAGS -c softmax.c -o softmax.o
$CC $CFLAGS -c conv2d.c -o conv2d.o

# ========== Compile Pure C Operators ==========
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/add.c -o add.o
$CC $CFLAGS -c c/svdf.c -o svdf.o
$CC $CFLAGS -c c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o

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

# ========== Release Mode: No Link, No Run ==========
echo "Release mode: All .o files generated"
echo "To link manually, execute:"
echo "  $CC $CFLAGS -T linker_arm.ld ... $CMSIS_NN_LIB ... -o model.elf"
