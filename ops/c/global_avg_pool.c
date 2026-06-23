#include "tinymlc.h"


void tmlc_global_avg_pool_s8(
    const int8_t* input,
    int8_t* output,
    int H, int W, int C,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t input_scale_q,  // scale * 2^15, Q15 format
    int32_t output_scale_q)  // scale * 2^15, Q15 format
{
    int total_pixels = H * W;

    for (int c = 0; c < C; c++) {
        int64_t sum = 0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (h * W + w) * C + c;
                sum += (int64_t)input[idx] - input_zero_point;
            }
        }

        // Average: sum / total_pixels, then scale by input_scale / output_scale
        // All integer: (sum * input_scale_q * inv_total) / output_scale_q
        int64_t avg = sum / total_pixels;
        int64_t tmp = avg * input_scale_q / output_scale_q;
        tmp += output_zero_point;

        if (tmp > 127) tmp = 127;
        if (tmp < -128) tmp = -128;

        output[c] = (int8_t)tmp;
    }
}

void tmlc_global_avg_pool_f32(
    const float* input,
    float* output,
    int H, int W, int C)
{
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

