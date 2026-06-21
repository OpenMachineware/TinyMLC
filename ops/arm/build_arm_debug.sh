#!/bin/bash
# ARM Debug Build Script (Pure C Reference Implementation)

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
SIM="qemu-system-arm"
ARCH="cortex-m4"
ABI="aapcs"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib \
    -ffreestanding -fno-omit-frame-pointer \
    -DTINYMLC_DEBUG -I./include -I./c -I."

# ========== Compile C Operators ==========
$CC $CFLAGS -c c/fc.c -o fc.o
$CC $CFLAGS -c c/softmax.c -o softmax.o
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/add.c -o add.o
$CC $CFLAGS -c c/svdf.c -o svdf.o
$CC $CFLAGS -c c/conv2d.c -o conv2d.o
$CC $CFLAGS -c c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o

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
    fc.o softmax.o reshape.o add.o svdf.o \
    conv2d.o max_pool2d.o depthwise_conv2d.o \
    relu.o avg_pool2d.o transpose.o pad.o mean.o \
    $LSTM_OBJ \
    model.o main_test.o \
    -o model.elf

# ========== Run ==========
$SIM -M mps2-an386 -nographic -semihosting -serial mon:stdio -kernel model.elf
