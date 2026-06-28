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

void tmlc_concat_s8(const int8_t** inputs, const int* sizes, int num_inputs,
                    int8_t* output)
{
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        // Manual copy, avoid using memcpy
        for (int j = 0; j < sizes[i]; j++) {
            output[offset + j] = inputs[i][j];
        }
        offset += sizes[i];
    }
}