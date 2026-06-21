#!/bin/bash
# RISC-V NMSIS-NN Release Build Script (no main_test.c generation, no run)

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="riscv-none-elf-gcc"
ARCH="rv32imac"
ABI="ilp32"

NMSIS_NN_INC="${NMSIS_NN_INC:-/opt/NMSIS-NN/Include}"
NMSIS_NN_LIB="${NMSIS_NN_LIB:-/opt/NMSIS-NN/Lib/libnmsisnn.a}"

CFLAGS="-march=$ARCH -mabi=$ABI -nostdlib -ffreestanding"
CFLAGS="$CFLAGS -fno-omit-frame-pointer -nostartfiles -nodefaultlibs"
CFLAGS="$CFLAGS -I./include -I./c -I. -I$NMSIS_NN_INC"

# ========== Compile RISC-V Accelerated Operators ==========
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

# ========== LSTM (Pure C, uses LUT) ==========
$CC $CFLAGS -c c/lstm.c -o lstm.o
$CC $CFLAGS -c lut.c -o lut.o
LSTM_OBJ="lstm.o lut.o"

# ========== Startup and Debug ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o

# ========== Release Mode: No Link, No Run ==========
echo "Release mode: All .o files generated"
echo "To link manually, execute:"
echo "  $CC -T linker.ld ... $NMSIS_NN_LIB ... -o model.elf"
