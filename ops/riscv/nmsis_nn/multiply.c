/* TinyMLC - Tiny Machine Learning Compiler
*
 * Copyright (c) 2026 Jia Liu & TinyMLC Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is part of TinyMLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * NMSIS-NN accelerated element-wise multiply operator
 *
 * Wraps riscv_elementwise_mul_s8 to match TinyMLC's tmlc_multiply_s8 interface.
 * For symmetric int8 quantization (zero_point=0), offsets are set to 0.
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

void tmlc_multiply_s8(const int8_t* input1, const int8_t* input2,
                       int8_t* output, int size)
{
    // Identity quantization parameters for symmetric int8 multiply
    const int32_t offset = 0;
    const int32_t mult = 1 << 15;  // 1.0 in Q0.15 format
    const int32_t shift = 15;

    riscv_elementwise_mul_s8(input1, input2,
                            offset, offset,
                            output,
                            offset, mult, shift,
                            -128, 127,
                            size);
}