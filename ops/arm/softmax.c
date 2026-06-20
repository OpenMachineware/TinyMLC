#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size)
{
    const int32_t num_rows = 1;
    const int32_t row_size = size;
    const int32_t mult = 1073741824;
    const int32_t shift = -6;
    const int32_t diff_min = -128;

    arm_softmax_s8(input, num_rows, row_size, mult, shift, diff_min, output);
}
