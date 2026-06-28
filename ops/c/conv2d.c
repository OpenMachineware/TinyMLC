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

/**
 * int8 quantized convolution operator
 *
 * Quantization formula:
 *   output = round((acc * multiplier) >> (31 + shift))
 *   where acc = sum(input * weight) + bias
 *
 * multiplier and shift are calculated by compiler based on
 * quantization params:
 *   effective_scale = (input_scale * weight_scale) /
 *       output_scale
 *   multiplier = effective_scale * 2^31 (adjusted to Q31 range)
 *   shift adjusts multiplier to valid range
 *
 * Parameter description:
 *   multiplier: Q31 fixed-point scale factor
 *   shift: right shift bits for scale precision
 *
 * Value range:
 *   127 / -128: int8 symmetric quantization range
 *   1 << 30: round-to-nearest offset (0.5 * 2^31)
 *   31: Q31 fixed-point format, highest precision for 32-bit
 *       signed int
 */
void tmlc_conv2d_s8(const int8_t* input,
                    const int8_t* weights,
                    const int32_t* bias,
                    int8_t* output,
                    int input_h, int input_w, int input_c,
                    int output_h, int output_w, int output_c,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int padding_h, int padding_w,
                    int32_t multiplier, int32_t shift)
{
    for (int oh = 0; oh < output_h; oh++) {
        for (int ow = 0; ow < output_w; ow++) {
            for (int oc = 0; oc < output_c; oc++) {
                int32_t sum = bias ? bias[oc] : 0;
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh * stride_h + kh - padding_h;
                        int iw = ow * stride_w + kw - padding_w;
                        if (ih >= 0 && ih < input_h &&
                            iw >= 0 && iw < input_w) {
                            for (int ic = 0; ic < input_c; ic++) {
                                int input_idx = ih * input_w * input_c +
                                                iw * input_c + ic;
                                int weight_idx = (
                                    oc * kernel_h * kernel_w * input_c
                                    + kh * kernel_w * input_c
                                    + kw * input_c + ic);
                                sum += (int32_t)input[input_idx] *
                                       (int32_t)weights[weight_idx];
                            }
                        }
                    }
                }
                // rescale: output = round((sum * multiplier) / 2^(31+shift))
                int64_t scaled = ((int64_t)sum * multiplier);
                // round-to-nearest
                scaled += (scaled >= 0) ? (1LL << 30) : -(1LL << 30);
                scaled >>= (31 + shift);
                if (scaled > 127) scaled = 127;   // int8 max
                if (scaled < -128) scaled = -128; // int8 min
                int out_idx = oh * output_w * output_c + ow * output_c + oc;
                output[out_idx] = (int8_t)scaled;
            }
        }
    }
}
