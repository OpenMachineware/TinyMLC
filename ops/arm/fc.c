#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size)
{
    // 量化参数暂时用 0，后续从模型提取
    const int32_t input_offset = 0;
    const int32_t filter_offset = 0;
    const int32_t output_offset = 0;
    const int32_t output_mult = 1073741824;  // 1.0 的 Q31 表示
    const int32_t output_shift = -6;          // 右移 6 位

    arm_fully_connected_s8(
        input,
        weights,
        input_size,
        output_size,
        filter_offset,
        input_offset,
        output_offset,
        bias,
        output,
        &output_mult,
        &output_shift
    );
}
