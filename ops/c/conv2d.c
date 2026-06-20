#include "tinymlc.h"

void tmlc_conv2d_s8(const int8_t* input,
                    const int8_t* weights,
                    const int32_t* bias,
                    int8_t* output,
                    int input_h, int input_w, int input_c,
                    int output_h, int output_w, int output_c,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int padding_h, int padding_w)
{
    // 对每个输出像素
    for (int oh = 0; oh < output_h; oh++) {
        for (int ow = 0; ow < output_w; ow++) {
            for (int oc = 0; oc < output_c; oc++) {
                int32_t sum = bias ? bias[oc] : 0;
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh * stride_h + kh - padding_h;
                        int iw = ow * stride_w + kw - padding_w;
                        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                            for (int ic = 0; ic < input_c; ic++) {
                                sum += (int32_t)input[ih * input_w * input_c + iw * input_c + ic]
                                     * (int32_t)weights[oc * kernel_h * kernel_w * input_c
                                            + kh * kernel_w * input_c + kw * input_c + ic];
                            }
                        }
                    }
                }
                output[oh * output_w * output_c + ow * output_c + oc] = (int8_t)(sum >> 8);
            }
        }
    }
}
