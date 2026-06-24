#include "tinymlc.h"
#include <math.h>


void tmlc_upsample_nearest_s8(
    const int8_t* input,
    int8_t* output,
    int H, int W, int C,
    int scale_h, int scale_w,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t input_scale_q,  // scale * 2^15, Q15 format
    int32_t output_scale_q)  // scale * 2^15, Q15 format
{
    int out_h = H * scale_h;
    int out_w = W * scale_w;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            int ih = oh / scale_h;
            int iw = ow / scale_w;
            for (int c = 0; c < C; c++) {
                int in_idx = (ih * W + iw) * C + c;
                int out_idx = (oh * out_w + ow) * C + c;
                output[out_idx] = input[in_idx];
            }
        }
    }
}
