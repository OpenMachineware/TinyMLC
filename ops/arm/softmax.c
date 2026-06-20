#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size)
{
    const int32_t input_mult = 1073741824;  // 1.0 的 Q31 表示
    const int32_t input_shift = -6;          // 右移 6 位
    const int32_t diff_min = INT32_MIN / 2;

    arm_softmax_s8(input, size, input_mult, input_shift, diff_min, output);
}
