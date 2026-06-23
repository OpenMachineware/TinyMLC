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
$CC $CFLAGS -c nmsis_nn/relu.c -o relu.o
$CC $CFLAGS -c nmsis_nn/relu6.c -o relu6.o
$CC $CFLAGS -c nmsis_nn/global_avg_pool.c -o global_avg_pool.o

# ========== Compile Pure C Operators (No NMSIS-NN acceleration) ==========
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/svdf.c -o svdf.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o
$CC $CFLAGS -c c/sigmoid.c -o sigmoid.o
$CC $CFLAGS -c c/tanh.c -o tanh.o
$CC $CFLAGS -c c/sub.c -o sub.o
$CC $CFLAGS -c c/concat.c -o concat.o
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
$CC -T link_riscv.ld -Wl,--no-dynamic-linker \
    start.o debug_print.o \
    fc.o softmax.o conv2d.o depthwise_conv2d.o \
    avg_pool2d.o max_pool2d.o add.o multiply.o \
    relu.o relu6.o global_avg_pool.o \
    reshape.o svdf.o transpose.o pad.o mean.o \
    sigmoid.o tanh.o sub.o concat.o \
    leaky_relu.o hard_sigmoid.o prelu.o clip.o \
    reduce_sum.o argmax.o flatten.o split.o strided_slice.o \
    nms.o upsample.o conv_transpose.o \
    $LSTM_OBJ \
    model.o main_test.o \
    $NMSIS_NN_LIB \
    -o model.elf

# ========== Run ==========
$SIM -M virt -nographic -bios none -kernel model.elf