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
#include <stddef.h>

void tmlc_pad_s8(const int8_t* input,
                 const int32_t* paddings,
                 int8_t* output,
                 int input_dims,
                 const int* input_shape,
                 const int* output_shape)
{
    // Simple implementation: only supports 4D padding
    // paddings format: [top, bottom, left, right, front, back, ...]
    // For 4D NHWC: [0, 0, top, bottom, left, right, 0, 0]

    if (input_dims != 4 || input == NULL ||
        output == NULL || paddings == NULL) {
        return;
    }

    int pad_top = paddings[2];
    int pad_bottom = paddings[3];
    int pad_left = paddings[4];
    int pad_right = paddings[5];

    int input_h = input_shape[1];
    int input_w = input_shape[2];
    int input_c = input_shape[3];

    int output_h = output_shape[1];
    int output_w = output_shape[2];
    int output_c = output_shape[3];

    // Initialize output to 0
    int output_size = output_h * output_w * output_c;
    for (int i = 0; i < output_size; i++) {
        output[i] = 0;
    }

    // Copy input to padded position
    for (int h = 0; h < input_h; h++) {
        for (int w = 0; w < input_w; w++) {
            for (int c = 0; c < input_c; c++) {
                int oh = h + pad_top;
                int ow = w + pad_left;
                output[oh * output_w * output_c + ow * output_c + c] =
                    input[h * input_w * input_c + w * input_c + c];
            }
        }
    }
}
