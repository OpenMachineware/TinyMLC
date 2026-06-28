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


void tmlc_split_s8(
    const int8_t* input,
    int8_t** outputs,
    int H, int W, int C,
    int* split_sizes,
    int num_splits,
    int axis)
{
    int offset = 0;

    for (int s = 0; s < num_splits; s++) {
        int size = split_sizes[s];

        if (axis == 3) {
            int bytes = H * W * size;
            for (int i = 0; i < bytes; i++) {
                outputs[s][i] = input[offset + i];
            }
            offset += bytes;
        } else if (axis == 1) {
            for (int h = 0; h < size; h++) {
                for (int w = 0; w < W; w++) {
                    for (int c = 0; c < C; c++) {
                        int in_idx = (offset + h) * W * C + w * C + c;
                        int out_idx = h * W * C + w * C + c;
                        outputs[s][out_idx] = input[in_idx];
                    }
                }
            }
            offset += size;
        }
    }
}
