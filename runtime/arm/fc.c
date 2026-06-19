#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size)
{
    arm_fully_connected_s8(
        input,
        weights,
        input_size,
        output_size,
        0,  // filter_offset
        0,  // input_offset
        0,  // output_offset
        bias,
        output,
        NULL  // output_mult
    );
}
