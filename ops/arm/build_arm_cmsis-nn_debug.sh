#!/bin/bash
# ARM CMSIS-NN Debug 构建脚本

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
SIM="qemu-system-arm"
ARCH="cortex-m4"
ABI="aapcs"

# CMSIS-NN 路径（用户需要修改或通过环境变量设置）
CMSIS_NN_INC="${CMSIS_NN_INC:-/opt/CMSIS-NN/Include}"
CMSIS_NN_LIB="${CMSIS_NN_LIB:-/opt/CMSIS-NN/Lib/libcmsisnn.a}"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib -ffreestanding -fno-omit-frame-pointer -DTINYMLC_DEBUG -I./include -I./c -I. -I$CMSIS_NN_INC"

cd tinymlc_generated/

# ========== 编译 ARM 加速算子 ==========
$CC $CFLAGS -c arm/op_fc.c -o op_fc.o
$CC $CFLAGS -c arm/op_softmax.c -o op_softmax.o
$CC $CFLAGS -c arm/op_conv2d.c -o op_conv2d.o

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
$CC $CFLAGS -T linker_arm.ld \
    start.o debug_print.o \
    op_fc.o op_softmax.o op_conv2d.o \
    op_reshape.o op_add.o op_svdf.o \
    op_max_pool2d.o op_depthwise_conv2d.o \
    op_relu.o op_avg_pool2d.o op_transpose.o op_pad.o op_mean.o \
    $LSTM_OBJ \
    model.o main_test.o \
    $CMSIS_NN_LIB \
    -o model.elf

# ========== 运行 ==========
$SIM -M mps2-an386 -nographic -semihosting -serial mon:stdio -kernel model.elf
