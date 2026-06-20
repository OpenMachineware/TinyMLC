#!/bin/bash
# ARM CMSIS-NN Release 构建脚本（不生成 main_test.c，不运行）

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
ARCH="cortex-m4"
ABI="aapcs"

CMSIS_NN_INC="${CMSIS_NN_INC:-/opt/CMSIS-NN/Include}"
CMSIS_NN_LIB="${CMSIS_NN_LIB:-/opt/CMSIS-NN/Lib/libcmsisnn.a}"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib -ffreestanding -fno-omit-frame-pointer -I./include -I./c -I. -I$CMSIS_NN_INC"

# ========== 编译 ARM 加速算子 ==========
$CC $CFLAGS -c arm/fc.c -o fc.o
$CC $CFLAGS -c arm/softmax.c -o softmax.o
$CC $CFLAGS -c arm/conv2d.c -o conv2d.o

# ========== 编译纯 C 算子 ==========
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
    $CC $CFLAGS -c lstm/lstm.c -o lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="lstm.o lut.o"
else
    LSTM_OBJ=""
fi

# ========== 启动和调试 ==========
$CC $CFLAGS -c start.S -o start.o
$CC $CFLAGS -c debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o

# ========== Release 模式：不链接，不运行 ==========
echo "Release 模式：已生成所有 .o 文件"
echo "如需链接，请手动执行："
echo "  $CC $CFLAGS -T linker_arm.ld ... $CMSIS_NN_LIB ... -o model.elf"
