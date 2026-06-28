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

#include "tinymlc.h"


void tmlc_hard_sigmoid_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int32_t zero_point,
    int32_t input_scale_q,   // Q15 format
    int32_t output_scale_q)  // Q15 format
{
    // 3.0 in Q15: 3 * 32768 = 98304
    // 6.0 in Q15: 6 * 32768 = 196608
    int32_t three_q15 = 98304;
    int32_t six_q15 = 196608;

    for (int i = 0; i < size; i++) {
        // Dequantize input to Q15: x * input_scale
        int32_t x = ((int32_t)input[i] - zero_point) * input_scale_q;

        // x + 3
        int32_t y = x + three_q15;

        // clamp to [0, 6]
        if (y < 0) y = 0;
        if (y > six_q15) y = six_q15;

        // Divide by 6: (y / 6) * (1 / output_scale)
        // y / 6 = y * (1/6) ≈ y / 6
        // Then quantize to output scale
        int32_t y_div_6 = y / 6;

        // Scale to output: y_div_6 / output_scale
        int32_t out = (y_div_6 * 32768) / output_scale_q;

        out += zero_point;

        if (out > 127) out = 127;
        if (out < -128) out = -128;

        output[i] = (int8_t)out;
    }
}
