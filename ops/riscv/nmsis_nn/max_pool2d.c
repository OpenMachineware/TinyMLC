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

/**
 * NMSIS-NN accelerated max pooling operator
 *
 * Wraps riscv_max_pool_s8 to match TinyMLC's tmlc_max_pool_2d_s8 interface.
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

static nmsis_nn_context ctx;
static int8_t nmsis_nn_buf[4096];

void tmlc_max_pool_2d_s8(const int8_t* input,
                         int8_t* output,
                         int input_h, int input_w, int input_c,
                         int output_h, int output_w, int output_c,
                         int pool_h, int pool_w,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w)
{
    nmsis_nn_pool_params pool_params = {
        .stride = {.h = stride_h, .w = stride_w},
        .padding = {.h = padding_h, .w = padding_w},
        .activation = {.min = -128, .max = 127}
    };

    nmsis_nn_dims input_dims = {
        .n = 1, .h = input_h, .w = input_w, .c = input_c};
    nmsis_nn_dims filter_dims = {
        .n = 1, .h = pool_h, .w = pool_w, .c = input_c};
    nmsis_nn_dims output_dims = {
        .n = 1, .h = output_h, .w = output_w, .c = output_c};

    ctx.buf = nmsis_nn_buf;
    ctx.size = sizeof(nmsis_nn_buf);

    riscv_max_pool_s8(&ctx, &pool_params,
                      &input_dims, input,
                      &filter_dims,
                      &output_dims, output);
}