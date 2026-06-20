#!/bin/bash
# 自动生成的 ARM 构建脚本

MODEL_PATH="${1:-trained_lstm_int8.tflite}"

CC="arm-none-eabi-gcc"
SIM="qemu-system-arm"
ARCH="cortex-m4"
ABI="aapcs"

CFLAGS="-mcpu=$ARCH -mthumb -mabi=$ABI -nostdlib -ffreestanding -fno-omit-frame-pointer -DTINYMLC_DEBUG -I../runtime/include -I../runtime/c -I../runtime -I../runtime/lstm/c -I."

cd tinymlc_generated/

# 算子
$CC $CFLAGS -c ../runtime/c/fc.c -o fc.o
$CC $CFLAGS -c ../runtime/c/softmax.c -o softmax.o
# ... 其他算子

# LSTM（按需）
if grep -q "HAS_LSTM" model_features.txt 2>/dev/null; then
    $CC $CFLAGS -c ../runtime/lstm/c/lstm.c -o lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="lstm.o lut.o"
else
    LSTM_OBJ=""
fi

# 启动和调试
$CC $CFLAGS -c ../runtime/arm/start.S -o start.o
$CC $CFLAGS -c ../runtime/arm/debug_print.c -o debug_print.o
$CC $CFLAGS -c model.c -o model.o
$CC $CFLAGS -c ../runtime/arm/main_test.c -o main_test.o

# 链接
$CC $CFLAGS -T ../runtime/arm/linker_arm.ld \
    start.o debug_print.o fc.o softmax.o $LSTM_OBJ model.o main_test.o \
    -o model.elf

$SIM -M mps2-an386 -nographic -semihosting -serial mon:stdio -kernel model.elf
