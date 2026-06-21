#!/bin/bash
# RISC-V NMSIS-NN Debug Build Script

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="riscv-none-elf-gcc"
SIM="qemu-system-riscv32"
ARCH="rv32imac"
ABI="ilp32"

# NMSIS-NN path
NMSIS_NN_INC="${NMSIS_NN_INC:-../../../third_party/NMSIS-1.6.0/Include}"
NMSIS_NN_LIB="${NMSIS_NN_LIB:-../../../third_party/NMSIS-1.6.0/Lib/libNMSISNN.a}"

CFLAGS="-march=$ARCH -mabi=$ABI -nostdlib -ffreestanding"
CFLAGS="$CFLAGS -fno-omit-frame-pointer -nostartfiles -nodefaultlibs"
CFLAGS="$CFLAGS -DTINYMLC_DEBUG -I./include -I./c -I. -I$NMSIS_NN_INC"

# ========== Compile NMSIS-NN Accelerated Operators ==========
$CC $CFLAGS -c nmsis_nn/fc.c -o fc.o
$CC $CFLAGS -c nmsis_nn/softmax.c -o softmax.o
$CC $CFLAGS -c nmsis_nn/conv2d.c -o conv2d.o
$CC $CFLAGS -c nmsis_nn/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c nmsis_nn/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c nmsis_nn/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c nmsis_nn/add.c -o add.o
$CC $CFLAGS -c nmsis_nn/multiply.c -o multiply.o

# ========== Compile Pure C Operators (No NMSIS-NN acceleration) ==========
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/svdf.c -o svdf.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o
$CC $CFLAGS -c c/sigmoid.c -o sigmoid.o
$CC $CFLAGS -c c/tanh.c -o tanh.o
$CC $CFLAGS -c c/sub.c -o sub.o
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
$CC -T link_riscv.ld -Wl,--no-dynamic-linker \
    start.o debug_print.o \
    fc.o softmax.o conv2d.o depthwise_conv2d.o \
    avg_pool2d.o max_pool2d.o add.o multiply.o \
    reshape.o svdf.o relu.o transpose.o pad.o mean.o \
    sigmoid.o tanh.o sub.o concat.o \
    $LSTM_OBJ \
    model.o main_test.o \
    $NMSIS_NN_LIB \
    -o model.elf

# ========== Run ==========
$SIM -M virt -nographic -bios none -kernel model.elf