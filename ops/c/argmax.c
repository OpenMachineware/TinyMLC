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


void tmlc_argmax_s8(
    const int8_t* input,
    int32_t* output,
    int H, int W, int C,
    int axis)
{
    if (axis == 3) {
        int total_pixels = H * W;
        for (int i = 0; i < total_pixels; i++) {
            int8_t max_val = input[i * C];
            int max_idx = 0;
            for (int c = 1; c < C; c++) {
                if (input[i * C + c] > max_val) {
                    max_val = input[i * C + c];
                    max_idx = c;
                }
            }
            output[i] = max_idx;
        }
    } else if (axis == 1) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int8_t max_val = input[w * C + c];
                int max_idx = 0;
                for (int h = 1; h < H; h++) {
                    int idx = h * W * C + w * C + c;
                    if (input[idx] > max_val) {
                        max_val = input[idx];
                        max_idx = h;
                    }
                }
                output[w * C + c] = max_idx;
            }
        }
    }
}
