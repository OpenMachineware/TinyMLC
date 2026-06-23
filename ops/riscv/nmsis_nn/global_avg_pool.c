/**
 * NMSIS-NN accelerated Global Average Pool operator
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

void tmlc_global_avg_pool_s8(const int8_t* input, int8_t* output,
                             int H, int W, int C,
                             int32_t input_zero_point,
                             int32_t output_zero_point,
                             int32_t input_scale_q, int32_t output_scale_q) {
    riscv_nn_pool_args pool_args;
    pool_args.padding.w = 0;
    pool_args.padding.h = 0;
    pool_args.stride.w = 1;
    pool_args.stride.h = 1;
    pool_args.ksize.w = W;
    pool_args.ksize.h = H;

    riscv_nn_per_channel_quant_params quant_params;
    riscv_nn_init_per_channel_quant_params(&quant_params, 1.0f);

    riscv_avgpool_s8(input, output, H, W, C, &pool_args, &quant_params);
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