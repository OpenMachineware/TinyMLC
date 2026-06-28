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

# For building CMSIS-NN
# cmake .. \
#     -DCMAKE_TOOLCHAIN_FILE=../arm-none-eabi-gcc.cmake \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/opt/cmsis-nn \
#     -DCMAKE_C_FLAGS="-mcpu=cortex-m4 -mthumb -mabi=aapcs \
#                      -mfloat-abi=soft -Ofast -DNDEBUG" \
#     -DCMAKE_CXX_FLAGS="-mcpu=cortex-m4 -mthumb -mabi=aapcs \
#                        -mfloat-abi=soft -Ofast -DNDEBUG"
# make
# Then copy Include and libcmsis-nn.a

# Set target system to bare-metal environment, processor architecture ARM
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

# Specify cross-compiler
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP arm-none-eabi-objdump)
set(CMAKE_SIZE arm-none-eabi-size)

# Disable CMake from trying to compile and run test programs
# (cross-compiled binaries cannot run on host)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Set path search rules to prevent accidentally
# including host libraries and headers
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
