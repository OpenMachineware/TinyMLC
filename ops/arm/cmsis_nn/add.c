/**
 * CMSIS-NN accelerated element-wise add operator
 *
 * Wraps arm_elementwise_add_s8 to match TinyMLC's tmlc_add_s8 interface.
 * For symmetric int8 quantization (zero_point=0), offsets and multipliers
 * are set to identity values.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_add_s8(const int8_t* input1, const int8_t* input2,
                 int8_t* output, int size)
{
    // Identity quantization parameters for symmetric int8
    const int32_t offset = 0;
    const int32_t mult = 1 << 15;  // 1.0 in Q0.15 format
    const int32_t shift = 15;
    const int32_t left_shift = 0;

    arm_elementwise_add_s8(input1, input2,
                          offset, mult, shift,  // input 1 params
                          offset, mult, shift,  // input 2 params
                          left_shift,
                          output,
                          offset, mult, shift,  // output params
                          -128, 127,           // activation clamp
                          size);
}
