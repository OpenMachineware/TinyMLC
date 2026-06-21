/**
 * CMSIS-NN accelerated softmax operator
 *
 * Wraps arm_softmax_s8 to match TinyMLC's tmlc_softmax_s8 interface.
 *
 * The softmax implementation uses a LUT-based approach for exponential
 * computation, compatible with TFLite micro.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size) {
    // CMSIS-NN softmax parameters for int8
    // mult and shift are based on the quantization scale
    // For typical int8 softmax, we use:
    //   mult = 1 << 30 (1.0 in Q31)
    //   shift = 0
    //   diff_min = -49 (to avoid numerical issues)
    //
    // The input is assumed to be in [-128, 127] range (int8 symmetric)
    // and output will be in [0, 127] range (probabilities)

    const int32_t mult = 1 << 30;
    const int32_t shift = 1;
    const int32_t diff_min = -49;

    arm_softmax_s8(input, 1, size, mult, shift, diff_min, output);
}
