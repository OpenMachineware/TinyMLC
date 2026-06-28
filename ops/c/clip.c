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


void tmlc_clip_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int8_t min_val,
    int8_t max_val,
    int32_t zero_point)
{
    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - zero_point;

        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;

        x += zero_point;
        if (x > 127) x = 127;
        if (x < -128) x = -128;
        output[i] = (int8_t)x;
    }
}
