/**
 * NMSIS-NN accelerated element-wise multiply operator
 *
 * Wraps riscv_elementwise_mul_s8 to match TinyMLC's tmlc_multiply_s8 interface.
 * For symmetric int8 quantization (zero_point=0), offsets are set to 0.
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

void tmlc_multiply_s8(const int8_t* input1, const int8_t* input2,
                       int8_t* output, int size)
{
    // Identity quantization parameters for symmetric int8 multiply
    const int32_t offset = 0;
    const int32_t mult = 1 << 15;  // 1.0 in Q0.15 format
    const int32_t shift = 15;

    riscv_elementwise_mul_s8(input1, input2,
                            offset, offset,
                            output,
                            offset, mult, shift,
                            -128, 127,
                            size);
}