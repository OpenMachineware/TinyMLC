#!/bin/bash
# RISC-V NMSIS-NN Debug 构建脚本

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="riscv-none-elf-gcc"
SIM="qemu-system-riscv32"
ARCH="rv32imac"
ABI="ilp32"

# NMSIS-NN 路径（用户需要修改或通过环境变量设置）
NMSIS_NN_INC="${NMSIS_NN_INC:-/opt/NMSIS-NN/Include}"
NMSIS_NN_LIB="${NMSIS_NN_LIB:-/opt/NMSIS-NN/Lib/libnmsisnn.a}"

CFLAGS="-march=$ARCH -mabi=$ABI -nostdlib -ffreestanding -fno-omit-frame-pointer -nostartfiles -nodefaultlibs -DTINYMLC_DEBUG -I./include -I./c -I. -I$NMSIS_NN_INC"

cd tinymlc_generated/

# ========== 编译 RISC-V 加速算子 ==========
$CC $CFLAGS -c riscv/op_fc.c -o op_fc.o
$CC $CFLAGS -c riscv/op_softmax.c -o op_softmax.o
$CC $CFLAGS -c riscv/op_conv2d.c -o op_conv2d.o

# ========== 编译纯 C 算子（未加速的） ==========
$CC $CFLAGS -c c/op_reshape.c -o op_reshape.o
$CC $CFLAGS -c c/op_add.c -o op_add.o
$CC $CFLAGS -c c/op_svdf.c -o op_svdf.o
$CC $CFLAGS -c c/op_max_pool2d.c -o op_max_pool2d.o
$CC $CFLAGS -c c/op_depthwise_conv2d.c -o op_depthwise_conv2d.o
$CC $CFLAGS -c c/op_relu.c -o op_relu.o
$CC $CFLAGS -c c/op_avg_pool2d.c -o op_avg_pool2d.o
$CC $CFLAGS -c c/op_transpose.c -o op_transpose.o
$CC $CFLAGS -c c/op_pad.c -o op_pad.o
$CC $CFLAGS -c c/op_mean.c -o op_mean.o

# ========== LSTM（按需） ==========
if grep -q "HAS_LSTM" model_features.txt 2>/dev/null; then
    $CC $CFLAGS -c lstm/op_lstm.c -o op_lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="op_lstm.o lut.o"
else
    LSTM_OBJ=""
fi

# ========== 启动和调试 ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o
$CC $CFLAGS -c main_test.c -o main_test.o

# ========== 链接 ==========
$CC -T linker.ld -Wl,--no-dynamic-linker \
    start.o debug_print.o \
    op_fc.o op_softmax.o op_conv2d.o \
    op_reshape.o op_add.o op_svdf.o \
    op_max_pool2d.o op_depthwise_conv2d.o \
    op_relu.o op_avg_pool2d.o op_transpose.o op_pad.o op_mean.o \
    $LSTM_OBJ \
    model.o main_test.o \
    $NMSIS_NN_LIB \
    -o model.elf

# ========== 运行 ==========
$SIM -M virt -nographic -bios none -kernel model.elf
