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


void tmlc_reduce_sum_s8(
    const int8_t* input,
    int32_t* output,
    int H, int W, int C,
    int axis,
    int32_t zero_point)
{
    if (axis == 3) {
        // Sum over channels
        int total_pixels = H * W;
        for (int i = 0; i < total_pixels; i++) {
            int64_t sum = 0;
            for (int c = 0; c < C; c++) {
                sum += (int64_t)input[i * C + c] - zero_point;
            }
            output[i] = (int32_t)sum;
        }
    } else if (axis == 1) {
        // Sum over height
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int64_t sum = 0;
                for (int h = 0; h < H; h++) {
                    int idx = (h * W + w) * C + c;
                    sum += (int64_t)input[idx] - zero_point;
                }
                output[w * C + c] = (int32_t)sum;
            }
        }
    } else if (axis == 2) {
        // Sum over width
        for (int h = 0; h < H; h++) {
            for (int c = 0; c < C; c++) {
                int64_t sum = 0;
                for (int w = 0; w < W; w++) {
                    int idx = (h * W + w) * C + c;
                    sum += (int64_t)input[idx] - zero_point;
                }
                output[h * C + c] = (int32_t)sum;
            }
        }
    }
}
