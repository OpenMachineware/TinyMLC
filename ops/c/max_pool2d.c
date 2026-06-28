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

void tmlc_max_pool_2d_s8(const int8_t* input,
                         int8_t* output,
                         int input_h, int input_w, int input_c,
                         int output_h, int output_w, int output_c,
                         int pool_h, int pool_w,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w)
{
    for (int oh = 0; oh < output_h; oh++) {
        for (int ow = 0; ow < output_w; ow++) {
            for (int oc = 0; oc < output_c; oc++) {
                int8_t max_val = -128;
                for (int ph = 0; ph < pool_h; ph++) {
                    for (int pw = 0; pw < pool_w; pw++) {
                        int ih = oh * stride_h + ph - padding_h;
                        int iw = ow * stride_w + pw - padding_w;
                        if (ih >= 0 && ih < input_h &&
                            iw >= 0 && iw < input_w) {
                            int8_t val = input[ih * input_w * input_c +
                                iw * input_c + oc];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                int out_idx = oh * output_w * output_c + ow * output_c + oc;
                output[out_idx] = max_val;
            }
        }
    }
}
