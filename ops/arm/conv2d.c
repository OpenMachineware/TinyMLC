#include "tinymlc.h"
#include "arm_nnfunctions.h"

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
    // 暂时用纯 C 实现，避免 CMSIS‑NN 参数复杂
    // 后续再改为 CMSIS‑NN 版本
    tmlc_conv2d_s8_ref(input, weights, bias, output,
                       input_h, input_w, input_c,
                       output_h, output_w, output_c,
                       kernel_h, kernel_w,
                       stride_h, stride_w,
                       padding_h, padding_w);
}
