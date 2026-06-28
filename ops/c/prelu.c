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


void tmlc_prelu_s8(
    const int8_t* input,
    const int8_t* alpha,
    int8_t* output,
    int H, int W, int C,
    int32_t zero_point)
{
    int total = H * W * C;

    for (int i = 0; i < total; i++) {
        int c = i % C;
        int32_t x = (int32_t)input[i] - zero_point;
        int32_t a = (int32_t)alpha[c];

        int32_t y;
        if (x > 0) {
            y = x;
        } else {
            // x * alpha (both are int8, product is int16)
            y = (x * a) / 128;  // assuming alpha is Q7 format
        }

        y += zero_point;
        if (y > 127) y = 127;
        if (y < -128) y = -128;
        output[i] = (int8_t)y;
    }
}
