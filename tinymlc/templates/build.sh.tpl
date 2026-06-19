#!/bin/bash
# 自动生成的构建脚本，请勿手动修改
# 由 tinymlc 自动生成

# 用法: ./build.sh <model_path>
# 示例: ./build.sh ../tflite-micro/.../micro_speech_lstm.tflite

RED="\033[31m"
YELLOW="\033[33m"
BLUE="\033[34m"
BOLD="\033[1m"
RESET="\033[0m"

MODEL_PATH="${1:-{{ model_path }}}"
TARGET="${2:-{{ target }}}"

echo -e "${BLUE}${BOLD}[INFO] 目标架构: $TARGET${RESET}"
echo -e "${BLUE}${BOLD}[INFO] 使用模型: $MODEL_PATH${RESET}"

# ========== 工具链配置 ==========
{% if target == "riscv" %}
CC="{{ cc }}"
SIM="{{ sim }}"
ARCH="{{ arch }}"
ABI="{{ abi }}"
{% elif target == "arm" %}
CC="{{ cc }}"
SIM="{{ sim }}"
ARCH="{{ arch }}"
ABI="{{ abi }}"
{% endif %}

# ========== 编译选项 ==========
CFLAGS_COMMON="-nostdlib -ffreestanding -fno-omit-frame-pointer -nostartfiles -nodefaultlibs"
{% if target == "riscv" %}
CFLAGS_ARCH="-march=$ARCH -mabi=$ABI"
{% elif target == "arm" %}
CFLAGS_ARCH="-mcpu=cortex-m4 -mthumb -mabi=$ABI"
{% endif %}
CFLAGS_DEBUG="-DTINYMLC_DEBUG"
CFLAGS_INC="-I../runtime/include -I../runtime/c -I../runtime -I../runtime/lstm/c -I."

CFLAGS="$CFLAGS_ARCH $CFLAGS_COMMON $CFLAGS_DEBUG $CFLAGS_INC"

# ========== 链接脚本 ==========
{% if target == "riscv" %}
LD_SCRIPT="../tinymlc/templates/link_riscv.ld"
{% elif target == "arm" %}
LD_SCRIPT="../tinymlc/templates/link_arm.ld"
{% endif %}

# ========== 编译 C 算子 ==========
$CC $CFLAGS -c ../runtime/c/fc.c -o fc.o
$CC $CFLAGS -c ../runtime/c/softmax.c -o softmax.o
$CC $CFLAGS -c ../runtime/c/reshape.c -o reshape.o
$CC $CFLAGS -c ../runtime/c/add.c -o add.o
$CC $CFLAGS -c ../runtime/c/svdf.c -o svdf.o
$CC $CFLAGS -c ../runtime/c/conv2d.c -o conv2d.o
$CC $CFLAGS -c ../runtime/c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c ../runtime/c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c ../runtime/c/relu.c -o relu.o
$CC $CFLAGS -c ../runtime/c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c ../runtime/c/transpose.c -o transpose.o
$CC $CFLAGS -c ../runtime/c/pad.c -o pad.o
$CC $CFLAGS -c ../runtime/c/mean.c -o mean.o

# ========== LSTM（按需） ==========
if grep -q "HAS_LSTM" model_features.txt 2>/dev/null; then
    echo -e "${BLUE}[INFO] 检测到 LSTM 算子，编译 lstm.c${RESET}"
    $CC $CFLAGS -c ../runtime/lstm/c/lstm.c -o lstm.o
    $CC $CFLAGS -c lut.c -o lut.o
    LSTM_OBJ="lstm.o lut.o"
else
    echo -e "${BLUE}[INFO] 未检测到 LSTM 算子，跳过 lstm.c${RESET}"
    LSTM_OBJ=""
fi

# ========== 启动代码 ==========
$CC $CFLAGS -c ../runtime/{{ target }}/start.S -o start.o

# ========== 辅助函数 ==========
$CC $CFLAGS -c debug_print.c -o debug_print.o

# ========== 模型和测试 ==========
$CC $CFLAGS -c model.c -o model.o
{% if mode == "debug" %}
$CC $CFLAGS -c main_test.c -o main_test.o
{% endif %}

{% if mode == "debug" %}
# ========== 链接 ==========
$CC -nostartfiles -T $LD_SCRIPT -Wl,--no-dynamic-linker \
    start.o debug_print.o \
    fc.o softmax.o reshape.o add.o svdf.o conv2d.o max_pool2d.o \
    depthwise_conv2d.o relu.o avg_pool2d.o transpose.o pad.o mean.o \
    $LSTM_OBJ \
    model.o main_test.o \
    -o model.elf

# ========== 模拟运行 ==========
    {% if target == "riscv" %}
    $SIM -M virt -nographic -bios none -kernel model.elf
    {% elif target == "arm" %}
    $SIM -M virt -nographic -kernel model.elf
    {% endif %}
{% else %}
echo -e "${BLUE}[INFO] Release 模式：已生成所有 .o 文件，跳过链接${RESET}"
echo -e "${BLUE}[INFO] 如需链接，请手动执行：${RESET}"
echo -e "${YELLOW}$CC -nostartfiles -T ../tinymlc/templates/link.ld -Wl,--no-dynamic-linker ... \${YOUR_OBJS} ... -o model.elf${RESET}"
{% endif %}
