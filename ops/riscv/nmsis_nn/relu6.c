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
 * NMSIS-NN accelerated ReLU6 operator
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"


void tmlc_relu6_s8(const int8_t* input, int8_t* output, int size,
                   int32_t zero_point, int32_t input_scale_q,
                   int32_t output_scale_q)
{
    static int8_t temp_buf[16384];

    if (size > 16384) {
        // Fallback: pure integer version
        int32_t six_q15 = 196608;  // 6.0 in Q15

        for (int i = 0; i < size; i++) {
            int32_t x = ((int32_t)input[i] - zero_point) * input_scale_q;
            if (x < 0) x = 0;
            if (x > six_q15) x = six_q15;
            int32_t out = (x * 32768) / output_scale_q;
            out += zero_point;
            if (out > 127) out = 127;
            if (out < -128) out = -128;
            output[i] = (int8_t)out;
        }
        return;
    }

    for (int i = 0; i < size; i++) {
        temp_buf[i] = input[i];
    }

    riscv_relu6_s8(temp_buf, size);

    for (int i = 0; i < size; i++) {
        output[i] = temp_buf[i];
    }
}
