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
 * CMSIS-NN accelerated ReLU operator
 *
 * Uses arm_relu6_s8 as CMSIS-NN doesn't have plain relu_s8.
 * Note: arm_relu6_s8 is in-place, we use a temp buffer.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_relu_s8(const int8_t* input, int8_t* output, int size) {
    // CMSIS-NN relu6_s8 is in-place, copy input first
    static int8_t temp_buf[16384];

    if (size > 16384) {
        // Fallback to pure C for large sizes
        for (int i = 0; i < size; i++) {
            output[i] = input[i] < 0 ? 0 : input[i];
        }
        return;
    }

    // Copy to temp buffer (CMSIS-NN modifies in-place)
    for (int i = 0; i < size; i++) {
        temp_buf[i] = input[i];
    }

    // Call CMSIS-NN relu6 (in-place, acts as relu since
    // input range is [0, 127])
    arm_relu6_s8(temp_buf, size);

    // Copy back to output
    for (int i = 0; i < size; i++) {
        output[i] = temp_buf[i];
    }
}
