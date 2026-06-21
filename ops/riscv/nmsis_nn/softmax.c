/**
 * NMSIS-NN accelerated softmax operator
 *
 * Wraps riscv_softmax_s8 to match TinyMLC's tmlc_softmax_s8 interface.
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size)
{
    riscv_softmax_s8(input, output, size);
}