#!/bin/bash

# -*- coding: utf-8 -*-
# TinyMLC - Tiny Machine Learning Compiler
#
# Copyright (c) 2026 Jia Liu & TinyMLC Contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of TinyMLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RISC-V Pure C Debug Build Script
# This script is copied to output directory and run from there

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${1:-model.onnx}"

export PATH="/opt/xpack-riscv-none-elf-gcc-15.2.0-1/bin:$PATH"

CC="riscv-none-elf-gcc"
LD="riscv-none-elf-ld"
SIM="qemu-system-riscv32"
ARCH="rv32imac_zicsr_zaamo_zalrsc"
ABI="ilp32"

CFLAGS="-march=$ARCH -mabi=$ABI -ffreestanding -mno-save-restore"
CFLAGS="$CFLAGS -fno-omit-frame-pointer"
CFLAGS="$CFLAGS -DTINYMLC_DEBUG -I./include -I./c -I."

# ========== Compile Pure C Operators ==========
$CC $CFLAGS -c c/fc.c -o fc.o
$CC $CFLAGS -c c/softmax.c -o softmax.o
$CC $CFLAGS -c c/conv2d.c -o conv2d.o
$CC $CFLAGS -c c/depthwise_conv2d.c -o depthwise_conv2d.o
$CC $CFLAGS -c c/avg_pool2d.c -o avg_pool2d.o
$CC $CFLAGS -c c/max_pool2d.c -o max_pool2d.o
$CC $CFLAGS -c c/global_avg_pool.c -o global_avg_pool.o
$CC $CFLAGS -c c/add.c -o add.o
$CC $CFLAGS -c c/multiply.c -o multiply.o
$CC $CFLAGS -c c/relu.c -o relu.o
$CC $CFLAGS -c c/relu6.c -o relu6.o
$CC $CFLAGS -c c/leaky_relu.c -o leaky_relu.o
$CC $CFLAGS -c c/hard_sigmoid.c -o hard_sigmoid.o
$CC $CFLAGS -c c/prelu.c -o prelu.o
$CC $CFLAGS -c c/clip.c -o clip.o
$CC $CFLAGS -c c/sigmoid.c -o sigmoid.o
$CC $CFLAGS -c c/tanh.c -o tanh.o
$CC $CFLAGS -c c/sub.c -o sub.o
$CC $CFLAGS -c c/concat.c -o concat.o
$CC $CFLAGS -c c/reshape.c -o reshape.o
$CC $CFLAGS -c c/transpose.c -o transpose.o
$CC $CFLAGS -c c/pad.c -o pad.o
$CC $CFLAGS -c c/mean.c -o mean.o
$CC $CFLAGS -c c/reduce_sum.c -o reduce_sum.o
$CC $CFLAGS -c c/argmax.c -o argmax.o
$CC $CFLAGS -c c/flatten.c -o flatten.o
$CC $CFLAGS -c c/split.c -o split.o
$CC $CFLAGS -c c/strided_slice.c -o strided_slice.o
$CC $CFLAGS -c c/nms.c -o nms.o
$CC $CFLAGS -c c/upsample.c -o upsample.o
$CC $CFLAGS -c c/conv_transpose.c -o conv_transpose.o
$CC $CFLAGS -c c/svdf.c -o svdf.o

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
# Use -nostartfiles to skip crt0.o and avoid _start multiple definition
# Use -nodefaultlibs to skip default libraries but still link gcc library
$CC $CFLAGS -nostartfiles -nodefaultlibs -s -T link_riscv.ld \
    start.o \
    debug_print.o \
    fc.o softmax.o conv2d.o depthwise_conv2d.o \
    avg_pool2d.o max_pool2d.o global_avg_pool.o add.o multiply.o \
    relu.o relu6.o leaky_relu.o hard_sigmoid.o prelu.o clip.o \
    sigmoid.o tanh.o sub.o concat.o reshape.o transpose.o pad.o mean.o \
    reduce_sum.o argmax.o flatten.o split.o strided_slice.o \
    nms.o upsample.o conv_transpose.o svdf.o \
    $LSTM_OBJ \
    model.o main_test.o \
    -o model.elf -lgcc

# ========== Run ==========
$SIM -M virt -nographic -bios none -kernel model.elf