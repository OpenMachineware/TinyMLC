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

// Precomputed tanh lookup table (int8 input -> int8 output)
// Input range: -128 to 127 (mapped to -1.0 to ~1.0)
// Output range: tanh(-1.0)~tanh(1.0) = -0.76~0.76
// Mapped to int8: tanh(x) * 127
static const int8_t tanh_lut[256] = {
    -97, -96, -95, -94, -93, -92, -91, -90,
    -89, -88, -87, -86, -85, -84, -83, -82,
    -81, -80, -79, -78, -77, -76, -75, -74,
    -73, -72, -71, -70, -69, -68, -67, -66,
    -65, -64, -63, -62, -61, -60, -59, -58,
    -57, -56, -55, -54, -53, -52, -51, -50,
    -49, -48, -47, -46, -45, -44, -43, -42,
    -41, -40, -39, -38, -37, -36, -35, -34,
    -33, -32, -31, -30, -29, -28, -27, -26,
    -25, -24, -23, -22, -21, -20, -19, -18,
    -17, -16, -15, -14, -13, -12, -11, -10,
     -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,
     -1,   0,   1,   2,   3,   4,   5,   6,
      7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,
     23,  24,  25,  26,  27,  28,  29,  30,
     31,  32,  33,  34,  35,  36,  37,  38,
     39,  40,  41,  42,  43,  44,  45,  46,
     47,  48,  49,  50,  51,  52,  53,  54,
     55,  56,  57,  58,  59,  60,  61,  62,
     63,  64,  65,  66,  67,  68,  69,  70,
     71,  72,  73,  74,  75,  76,  77,  78,
     79,  80,  81,  82,  83,  84,  85,  86,
     87,  88,  89,  90,  91,  92,  93,  94,
     95,  96,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
     97,  97,  97,  97,  97,  97,  97,  97,
};

void tmlc_tanh_s8(const int8_t* input, int8_t* output, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] = tanh_lut[(uint8_t)input[i]];
    }
}