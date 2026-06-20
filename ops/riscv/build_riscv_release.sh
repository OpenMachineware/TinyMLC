#!/bin/bash
# RISC-V Release 构建脚本（纯 C 参考实现，不生成 main_test.c，不运行）

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="riscv-none-elf-gcc"
ARCH="rv32imac"
ABI="ilp32"

CFLAGS="-march=$ARCH -mabi=$ABI -nostdlib -ffreestanding -fno-omit-frame-pointer -nostartfiles -nodefaultlibs -I./include -I./c -I."

# ========== 编译 C 算子 ==========
$CC $CFLAGS -c c/op_fc.c -o op_fc.o
$CC $CFLAGS -c c/op_softmax.c -o op_softmax.o
$CC $CFLAGS -c c/op_reshape.c -o op_reshape.o
$CC $CFLAGS -c c/op_add.c -o op_add.o
$CC $CFLAGS -c c/op_svdf.c -o op_svdf.o
$CC $CFLAGS -c c/op_conv2d.c -o op_conv2d.o
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

# ========== Release 模式：不链接，不运行 ==========
echo "Release 模式：已生成所有 .o 文件"
echo "如需链接，请手动执行："
echo "  $CC -T linker.ld start.o debug_print.o ... -o model.elf"
