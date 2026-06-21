#include "tinymlc.h"

/**
 * int8 量化卷积算子
 * 
 * 量化公式:
 *   output = round((acc * multiplier) >> (31 + shift))
 *   其中 acc = sum(input * weight) + bias
 * 
 * multiplier 和 shift 由编译器根据量化参数计算:
 *   effective_scale = (input_scale * weight_scale) / output_scale
 *   multiplier = effective_scale * 2^31 (调整到 Q31 范围)
 *   shift 用于调整 multiplier 到有效范围
 * 
 * 参数说明:
 *   multiplier: Q31 定点数表示的缩放因子
 *   shift: 右移位数，用于调整缩放精度
 * 
 * 数值范围:
 *   127 / -128: int8 对称量化的数值范围
 *   1 << 30: round-to-nearest 的偏移量 (0.5 * 2^31)
 *   31: Q31 定点数格式，32位有符号数的最高精度表示
 */
void tmlc_conv2d_s8(const int8_t* input,
                    const int8_t* weights,
                    const int32_t* bias,
                    int8_t* output,
                    int input_h, int input_w, int input_c,
                    int output_h, int output_w, int output_c,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int padding_h, int padding_w,
                    int32_t multiplier, int32_t shift)
{
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
                // rescale: output = round((sum * multiplier) / 2^(31+shift))
                int64_t scaled = ((int64_t)sum * multiplier);
                scaled += (scaled >= 0) ? (1LL << 30) : -(1LL << 30);  // round-to-nearest
                scaled >>= (31 + shift);
                if (scaled > 127) scaled = 127;   // int8 max
                if (scaled < -128) scaled = -128; // int8 min
                output[oh * output_w * output_c + ow * output_c + oc] = (int8_t)scaled;
            }
        }
    }
}
