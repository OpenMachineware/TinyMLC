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
 * CMSIS-NN accelerated element-wise add operator
 *
 * Wraps arm_elementwise_add_s8 to match TinyMLC's tmlc_add_s8 interface.
 * For symmetric int8 quantization (zero_point=0), offsets and multipliers
 * are set to identity values.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_add_s8(const int8_t* input1, const int8_t* input2,
                 int8_t* output, int size)
{
    // Identity quantization parameters for symmetric int8
    const int32_t offset = 0;
    const int32_t mult = 1 << 15;  // 1.0 in Q0.15 format
    const int32_t shift = 15;
    const int32_t left_shift = 0;

    arm_elementwise_add_s8(input1, input2,
                          offset, mult, shift,  // input 1 params
                          offset, mult, shift,  // input 2 params
                          left_shift,
                          output,
                          offset, mult, shift,  // output params
                          -128, 127,           // activation clamp
                          size);
}
