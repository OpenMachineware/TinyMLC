/**
 * CMSIS-NN accelerated ReLU operator
 *
 * Uses arm_relu6_s8 as CMSIS-NN doesn't have plain relu_s8.
 * Note: arm_relu6_s8 is in-place, we use a temp buffer.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_relu_s8(const int8_t* input, int8_t* output, int size) {
    // CMSIS-NN relu6_s8 is in-place, copy input first
    static int8_t temp_buf[16384];

    if (size > 16384) {
        // Fallback to pure C for large sizes
        for (int i = 0; i < size; i++) {
            output[i] = input[i] < 0 ? 0 : input[i];
        }
        return;
    }

    // Copy to temp buffer (CMSIS-NN modifies in-place)
    for (int i = 0; i < size; i++) {
        temp_buf[i] = input[i];
    }

    // Call CMSIS-NN relu6 (in-place, acts as relu since
    // input range is [0, 127])
    arm_relu6_s8(temp_buf, size);

    // Copy back to output
    for (int i = 0; i < size; i++) {
        output[i] = temp_buf[i];
    }
}
