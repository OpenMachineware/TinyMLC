#include "tinymlc.h"


void tmlc_conv_transpose_s8(
    const int8_t* input,
    const int8_t* weights,
    const int32_t* bias,
    int8_t* output,
    int in_h, int in_w, int in_c,
    int out_h, int out_w, int out_c,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int32_t multiplier,
    int32_t shift,
    int32_t input_zero_point,
    int32_t output_zero_point)
{
    int out_size = out_h * out_w * out_c;

    // Zero initialize output (manual loop)
    for (int i = 0; i < out_size; i++) {
        output[i] = 0;
    }

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                int32_t acc = 0;
                if (bias) {
                    acc = bias[oc];
                }

                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh - kh + pad_h;
                        int iw = ow - kw + pad_w;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            for (int ic = 0; ic < in_c; ic++) {
                                int in_idx = (ih * in_w + iw) * in_c + ic;
                                int w_idx = ((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                                int32_t in_val = (int32_t)input[in_idx] - input_zero_point;
                                int32_t w_val = (int32_t)weights[w_idx];
                                acc += in_val * w_val;
                            }
                        }
                    }
                }

                // Quantization: multiply then shift
                int64_t tmp = (int64_t)acc * multiplier;
                if (shift > 0) {
                    tmp = (tmp + (1 << (shift - 1))) >> shift;
                }
                tmp += output_zero_point;

                if (tmp > 127) tmp = 127;
                if (tmp < -128) tmp = -128;

                int out_idx = (oh * out_w + ow) * out_c + oc;
                output[out_idx] = (int8_t)tmp;
            }
        }
    }
}
