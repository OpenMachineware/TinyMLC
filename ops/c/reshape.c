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

void tmlc_reshape_s8(const int8_t* input, int8_t* output,
                     int input_size, const int* new_shape, int shape_size)
{
    int output_size = 1;
    for (int i = 0; i < shape_size; i++) {
        output_size *= new_shape[i];
    }

    if (input_size != output_size) return;

    for (int i = 0; i < input_size; i++) {
        output[i] = input[i];
    }
}
