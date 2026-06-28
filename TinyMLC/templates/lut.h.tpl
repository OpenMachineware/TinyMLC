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

// Auto-generated LUT tables, do not modify manually
// sigmoid and tanh int16 lookup tables (256 entries)

#ifndef TINYMLC_LUT_H
#define TINYMLC_LUT_H

#include <stdint.h>

// Sigmoid LUT (input range [-8,8), output range [0,1] quantized to [0,32767])
static const int16_t sigmoid_lut[256] = {
{% for val in sigmoid_lut %}
    {{ val }}{% if not loop.last %},{% endif %}
{% endfor %}
};

// Tanh LUT (input [-8,8), output [-1,1] quantized to [-32768,32767])
static const int16_t tanh_lut[256] = {
{% for val in tanh_lut %}
    {{ val }}{% if not loop.last %},{% endif %}
{% endfor %}
};

// LUT lookup functions (linear interpolation)
static inline int16_t sigmoid_lut_lookup(int32_t x) {
    // x is int32 accumulator, map to [-8,8) range based on quantization
    // Assume x is already scaled to [0, 256*8) range
    int32_t idx = (x >> 8) & 0xFF;
    uint8_t frac = (uint8_t)(x & 0xFF);

    int16_t lower = sigmoid_lut[idx];
    int16_t upper = sigmoid_lut[idx + 1];

    int32_t result = lower + ((upper - lower) * frac >> 8);
    return (int16_t)result;
}

static inline int16_t tanh_lut_lookup(int32_t x) {
    int32_t idx = (x >> 8) & 0xFF;
    uint8_t frac = (uint8_t)(x & 0xFF);

    int16_t lower = tanh_lut[idx];
    int16_t upper = tanh_lut[idx + 1];

    int32_t result = lower + ((upper - lower) * frac >> 8);
    return (int16_t)result;
}

#endif // TINYMLC_LUT_H
