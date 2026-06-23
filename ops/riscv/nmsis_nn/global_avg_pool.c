/**
 * NMSIS-NN accelerated Global Average Pool operator
 * Updated for NMSIS 1.6.0 API
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

void tmlc_global_avg_pool_s8(const int8_t* input, int8_t* output,
                             int H, int W, int C,
                             int32_t input_zero_point,
                             int32_t output_zero_point,
                             int32_t input_scale_q, int32_t output_scale_q) {
    // NMSIS 1.6.0 requires context structure
    nmsis_nn_context ctx;
    int32_t buffer_size = riscv_avgpool_s8_get_buffer_size(1, C);
    int8_t buffer[buffer_size];
    ctx.buf = buffer;
    ctx.size = buffer_size;

    // Pool parameters
    nmsis_nn_pool_params pool_params;
    pool_params.padding.h = 0;
    pool_params.padding.w = 0;
    pool_params.stride.h = 1;
    pool_params.stride.w = 1;
    pool_params.ksize.h = H;
    pool_params.ksize.w = W;
    pool_params.activation.min = -128;
    pool_params.activation.max = 127;

    // Input dimensions [batch, height, width, channels]
    nmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = H;
    input_dims.w = W;
    input_dims.c = C;

    // Filter dimensions (same as input for global pool)
    nmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = H;
    filter_dims.w = W;
    filter_dims.c = C;

    // Output dimensions [batch, 1, 1, channels]
    nmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = C;

    riscv_avgpool_s8(&ctx, &pool_params, &input_dims, input,
                     &filter_dims, &output_dims, output);
}

void tmlc_global_avg_pool_f32(const float* input, float* output,
                              int H, int W, int C) {
    int total_pixels = H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (h * W + w) * C + c;
                sum += input[idx];
            }
        }
        output[c] = sum / total_pixels;
    }
}