/**
 * NMSIS-NN accelerated ReLU operator
 *
 * Uses riscv_relu6_s8 as NMSIS-NN doesn't have plain relu_s8.
 * Note: riscv_relu6_s8 is in-place, we use a temp buffer.
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

void tmlc_relu_s8(const int8_t* input, int8_t* output, int size) {
    static int8_t temp_buf[16384];

    if (size > 16384) {
        for (int i = 0; i < size; i++) {
            output[i] = input[i] < 0 ? 0 : input[i];
        }
        return;
    }

    for (int i = 0; i < size; i++) {
        temp_buf[i] = input[i];
    }

    riscv_relu6_s8(temp_buf, size);

    for (int i = 0; i < size; i++) {
        output[i] = temp_buf[i];
    }
}
