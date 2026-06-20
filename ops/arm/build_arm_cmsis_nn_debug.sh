#!/bin/bash
# ARM CMSIS-NN Debug 构建脚本

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
SIM="qemu-system-arm"
ARCH="cortex-m4"
ABI="aapcs"
CMSIS_NN_INC="../third_party/CMSIS-NN-5.0.0/Include"
CMSIS_NN_LIB="../third_party/CMSIS-NN-5.0.0/Lib/libcmsis-nn.a"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib -ffreestanding -fno-omit-frame-pointer -DTINYMLC_DEBUG -I./include -I./c -I. -I$CMSIS_NN_INC"
LDFLAGS="-L$(dirname $CMSIS_NN_LIB) -lcmsis-nn -lgcc"

# ========== 编译 ARM 加速算子 ==========
$CC $CFLAGS -c fc.c -o fc.o
$CC $CFLAGS -c softmax.c -o softmax.o
$CC $CFLAGS -c conv2d.c -o conv2d.o

# ========== 编译纯 C 算子（未加速的） ==========
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

# ========== LSTM（按需） ==========
if grep -q "HAS_LSTM" model_features.txt 2>/dev/null; then
    $CC $CFLAGS -c lstm/c/lstm.c -o lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="lstm.o lut.o"
else
    LSTM_OBJ=""
fi

# ========== 启动和调试 ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o
$CC $CFLAGS -c main_test.c -o main_test.o

# ========== 链接 ==========
$CC $CFLAGS -T link_arm.ld \
    start.o debug_print.o \
    fc.o softmax.o conv2d.o \
    reshape.o add.o svdf.o \
    max_pool2d.o depthwise_conv2d.o \
    relu.o avg_pool2d.o transpose.o pad.o mean.o \
    $LSTM_OBJ \
    model.o main_test.o \
    $LDFLAGS \
    -o model.elf

# ========== 运行 ==========
$SIM -M mps2-an386 -nographic -semihosting -serial mon:stdio -kernel model.elf
