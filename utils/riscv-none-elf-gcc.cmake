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

# For building NMSIS-NN
# cmake ../Source \
#   -DCMAKE_TOOLCHAIN_FILE=../riscv-none-elf-gcc.cmake \
#   -DRISCV_ARCH=rv32imac \
#   -DRISCV_ABI=ilp32 \
#   -DRISCV_MODEL=medany
# make
# Then copy Include and libNMSISNN.a

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)

set(CROSS_COMPILE "riscv-none-elf-")

set(CMAKE_C_COMPILER ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE}g++)
set(CMAKE_ASM_COMPILER ${CROSS_COMPILE}gcc)
set(CMAKE_AR ${CROSS_COMPILE}ar)
set(CMAKE_OBJCOPY ${CROSS_COMPILE}objcopy)
set(CMAKE_OBJDUMP ${CROSS_COMPILE}objdump)

set(RISCV_ARCH "rv32imac")
set(RISCV_ABI "ilp32")

add_compile_options(
    -march=${RISCV_ARCH}
    -mabi=${RISCV_ABI}
    -mcmodel=medany
    -O2
    -Wall
)
