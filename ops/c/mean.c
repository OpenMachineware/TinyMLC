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

void tmlc_mean_s8(const int8_t* input,
                  int8_t* output,
                  int input_dims,
                  const int* input_shape,
                  const int* output_shape,
                  const int32_t* axis,
                  int axis_count,
                  int keep_dims)
{
    // Simple implementation: global average (all dimensions)
    // For MobileNetV2 MEAN, typically global average pooling
    // Input shape: [1, 7, 7, 1280] -> Output: [1, 1, 1, 1280]

    if (input == NULL || output == NULL || input_dims <= 0) {
        return;
    }

    // Calculate total element count
    int total_size = 1;
    for (int i = 0; i < input_dims; i++) {
        total_size *= input_shape[i];
    }

    // If output dimension is 1, do global average
    int output_size = 1;
    for (int i = 0; i < input_dims; i++) {
        output_size *= output_shape[i];
    }

    if (output_size == 1) {
        // Global average
        int32_t sum = 0;
        for (int i = 0; i < total_size; i++) {
            sum += input[i];
        }
        output[0] = (int8_t)(sum / total_size);
    } else {
        // Average along last dimension (most common case)
        int last_dim = input_shape[input_dims - 1];
        int num_groups = total_size / last_dim;

        for (int g = 0; g < num_groups; g++) {
            int32_t sum = 0;
            for (int d = 0; d < last_dim; d++) {
                sum += input[g * last_dim + d];
            }
            output[g] = (int8_t)(sum / last_dim);
        }
    }
}
