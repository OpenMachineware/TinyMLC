#!/bin/bash
# ARM CMSIS-NN Debug Build Script
# Auto-generated from template - DO NOT EDIT

cd "$(dirname "$0")"

MODEL_PATH="${1:-model.onnx}"

CC="arm-none-eabi-gcc"
SIM="qemu-system-arm"
ARCH="cortex-m4"
ABI="aapcs"
FPU="fpv4-sp-d16"
FLOAT_ABI="soft"

# CMSIS-NN paths (passed from codegen.py)
CMSIS_NN_INC="{{ accel_lib_inc }}"
CMSIS_NN_LIB="{{ accel_lib_lib }}"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -mfpu=$FPU -mfloat-abi=$FLOAT_ABI -nostdlib \
    -ffreestanding -fno-omit-frame-pointer \
    -DTINYMLC_DEBUG -I./include -I./c -I. -I$CMSIS_NN_INC"

# ========== CMSIS-NN Accelerated Operators ==========
$CC $CFLAGS -c c/fc.c -o fc.o
$CC $CFLAGS -c c/conv2d.c -o conv2d.o
$CC $CFLAGS -c c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c c/softmax.c -o softmax.o
$CC $CFLAGS -c c/add.c -o add.o
$CC $CFLAGS -c c/multiply.c -o multiply.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/relu6.c -o relu6.o
$CC $CFLAGS -c c/global_avg_pool.c -o global_avg_pool.o

# ========== Pure C Operators (No CMSIS-NN acceleration) ==========
$CC $CFLAGS -c c/sigmoid.c -o sigmoid.o
$CC $CFLAGS -c c/tanh.c -o tanh.o
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/concat.c -o concat.o
$CC $CFLAGS -c c/sub.c -o sub.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o
$CC $CFLAGS -c c/svdf.c -o svdf.o
$CC $CFLAGS -c c/leaky_relu.c -o leaky_relu.o
$CC $CFLAGS -c c/hard_sigmoid.c -o hard_sigmoid.o
$CC $CFLAGS -c c/prelu.c -o prelu.o
$CC $CFLAGS -c c/clip.c -o clip.o
$CC $CFLAGS -c c/reduce_sum.c -o reduce_sum.o
$CC $CFLAGS -c c/argmax.c -o argmax.o
$CC $CFLAGS -c c/flatten.c -o flatten.o
$CC $CFLAGS -c c/split.c -o split.o
$CC $CFLAGS -c c/strided_slice.c -o strided_slice.o
$CC $CFLAGS -c c/nms.c -o nms.o
$CC $CFLAGS -c c/upsample.c -o upsample.o
$CC $CFLAGS -c c/conv_transpose.c -o conv_transpose.o

# ========== LSTM (Pure C, uses LUT) ==========
$CC $CFLAGS -c c/lstm.c -o lstm.o
$CC $CFLAGS -c lut.c -o lut.o
LSTM_OBJ="lstm.o lut.o"

# ========== Startup and Debug ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o
$CC $CFLAGS -c main_test.c -o main_test.o

# ========== Link ==========
$CC $CFLAGS -T link_arm.ld \
    start.o debug_print.o \
    fc.o conv2d.o depthwise_conv2d.o avg_pool2d.o max_pool2d.o \
    softmax.o add.o multiply.o relu.o relu6.o global_avg_pool.o \
    sigmoid.o tanh.o reshape.o concat.o sub.o \
    transpose.o pad.o mean.o svdf.o \
    leaky_relu.o hard_sigmoid.o prelu.o clip.o \
    reduce_sum.o argmax.o flatten.o split.o strided_slice.o \
    nms.o upsample.o conv_transpose.o \
    $LSTM_OBJ \
    model.o main_test.o \
    -L$(dirname $CMSIS_NN_LIB) -lcmsis-nn -lgcc -lm -o model.elf

# ========== Run ==========
$SIM -M mps2-an386 -nographic -semihosting-config enable=on,target=native -serial mon:stdio -kernel model.elf