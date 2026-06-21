/**
 * CMSIS-NN accelerated max pooling operator
 *
 * Wraps arm_max_pool_s8 to match TinyMLC's tmlc_max_pool_2d_s8 interface.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

static cmsis_nn_context ctx;
static int8_t cmsis_nn_buf[4096];

void tmlc_max_pool_2d_s8(const int8_t* input,
                         int8_t* output,
                         int input_h, int input_w, int input_c,
                         int output_h, int output_w, int output_c,
                         int pool_h, int pool_w,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w)
{
    cmsis_nn_pool_params pool_params = {
        .stride = {.h = stride_h, .w = stride_w},
        .padding = {.h = padding_h, .w = padding_w},
        .activation = {.min = -128, .max = 127}
    };

    cmsis_nn_dims input_dims = {.n = 1, .h = input_h, .w = input_w, .c = input_c};
    cmsis_nn_dims filter_dims = {.n = 1, .h = pool_h, .w = pool_w, .c = input_c};
    cmsis_nn_dims output_dims = {.n = 1, .h = output_h, .w = output_w, .c = output_c};

    ctx.buf = cmsis_nn_buf;
    ctx.size = sizeof(cmsis_nn_buf);

    arm_max_pool_s8(&ctx, &pool_params,
                    &input_dims, input,
                    &filter_dims,
                    &output_dims, output);
}
