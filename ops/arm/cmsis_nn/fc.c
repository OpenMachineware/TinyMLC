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
 * CMSIS-NN accelerated fully connected operator
 *
 * Wraps arm_fully_connected_s8 to match TinyMLC's tmlc_fully_connected_s8
 * interface.
 *
 * Note: CMSIS-NN requires input_zero_point and output_zero_point for
 * quantized operations. We use 0 for symmetric quantization.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

// CMSIS-NN context for optimized operations (scratch buffer)
static cmsis_nn_context ctx;

// Buffer for CMSIS-NN internal operations
static int8_t cmsis_nn_buf[8192];

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size,
                              int32_t multiplier,
                              int32_t shift)
{
    cmsis_nn_fc_params fc_params = {
        .input_offset = 0,
        .filter_offset = 0,
        .output_offset = 0,
        .activation = {.min = -128, .max = 127}
    };

    cmsis_nn_per_tensor_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift
    };

    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 1, .c = input_size};
    cmsis_nn_dims filter_dims = {
        .n = input_size, .h = 1, .w = 1, .c = output_size};
    cmsis_nn_dims bias_dims = {.n = output_size, .h = 1, .w = 1, .c = 1};
    cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = 1, .c = output_size};

    ctx.buf = cmsis_nn_buf;
    ctx.size = sizeof(cmsis_nn_buf);

    arm_fully_connected_s8(&ctx, &fc_params, &quant_params,
                            &input_dims, input,
                            &filter_dims, weights,
                            &bias_dims, bias,
                            &output_dims, output);
}
