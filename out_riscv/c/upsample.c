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

void tmlc_upsample_bilinear_f32(
    const float* input,
    float* output,
    int H, int W, int C,
    int scale_h, int scale_w)
{
    int out_h = H * scale_h;
    int out_w = W * scale_w;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            float ih_f = (float)oh / scale_h;
            float iw_f = (float)ow / scale_w;

            int ih0 = (int)ih_f;
            int iw0 = (int)iw_f;
            int ih1 = ih0 + 1;
            int iw1 = iw0 + 1;

            float dh = ih_f - ih0;
            float dw = iw_f - iw0;

            // Clamp to edges
            if (ih0 < 0) ih0 = 0;
            if (ih1 >= H) ih1 = H - 1;
            if (iw0 < 0) iw0 = 0;
            if (iw1 >= W) iw1 = W - 1;

            for (int c = 0; c < C; c++) {
                float v00 = input[(ih0 * W + iw0) * C + c];
                float v01 = input[(ih0 * W + iw1) * C + c];
                float v10 = input[(ih1 * W + iw0) * C + c];
                float v11 = input[(ih1 * W + iw1) * C + c];

                float v0 = v00 * (1.0f - dw) + v01 * dw;
                float v1 = v10 * (1.0f - dw) + v11 * dw;
                float val = v0 * (1.0f - dh) + v1 * dh;

                int out_idx = (oh * out_w + ow) * C + c;
                output[out_idx] = val;
            }
        }
    }
}

void tmlc_upsample_bilinear_s8(
    const int8_t* input,
    int8_t* output,
    int H, int W, int C,
    int scale_h, int scale_w,
    int32_t input_zero_point,
    int32_t output_zero_point,
    float input_scale,
    float output_scale)
{
    int out_h = H * scale_h;
    int out_w = W * scale_w;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            float ih_f = (float)oh / scale_h;
            float iw_f = (float)ow / scale_w;

            int ih0 = (int)ih_f;
            int iw0 = (int)iw_f;
            int ih1 = ih0 + 1;
            int iw1 = iw0 + 1;

            float dh = ih_f - ih0;
            float dw = iw_f - iw0;

            if (ih0 < 0) ih0 = 0;
            if (ih1 >= H) ih1 = H - 1;
            if (iw0 < 0) iw0 = 0;
            if (iw1 >= W) iw1 = W - 1;

            for (int c = 0; c < C; c++) {
                float f00 = ((float)input[(ih0 * W + iw0) * C + c] - input_zero_point) * input_scale;
                float f01 = ((float)input[(ih0 * W + iw1) * C + c] - input_zero_point) * input_scale;
                float f10 = ((float)input[(ih1 * W + iw0) * C + c] - input_zero_point) * input_scale;
                float f11 = ((float)input[(ih1 * W + iw1) * C + c] - input_zero_point) * input_scale;

                float v0 = f00 * (1.0f - dw) + f01 * dw;
                float v1 = f10 * (1.0f - dw) + f11 * dw;
                float val = v0 * (1.0f - dh) + v1 * dh;

                int qval = (int)(val / output_scale + output_zero_point);
                if (qval > 127) qval = 127;
                if (qval < -128) qval = -128;

                int out_idx = (oh * out_w + ow) * C + c;
                output[out_idx] = (int8_t)qval;
            }
        }
    }
}
