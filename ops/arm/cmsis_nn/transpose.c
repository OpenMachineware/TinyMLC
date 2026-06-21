/**
 * Transpose operator
 *
 * No CMSIS-NN acceleration available, uses simple implementation.
 */

#include <stddef.h>
#include "tinymlc.h"

static int get_position(const int* idx, const int* strides, int dims) {
    int pos = 0;
    for (int i = 0; i < dims; i++) {
        pos += idx[i] * strides[i];
    }
    return pos;
}

void tmlc_transpose_s8(const int8_t* input,
                       const int32_t* perm,
                       int8_t* output,
                       int input_dims,
                       const int* input_shape)
{
    if (input_dims <= 0 || input == NULL || output == NULL || perm == NULL) {
        return;
    }

    int output_shape[4];
    for (int i = 0; i < input_dims; i++) {
        output_shape[i] = input_shape[perm[i]];
    }

    int input_strides[4];
    input_strides[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; i--) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    int output_strides[4];
    output_strides[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; i--) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    int idx[4] = {0, 0, 0, 0};
    int total_output = 1;
    for (int i = 0; i < input_dims; i++) {
        total_output *= output_shape[i];
    }

    for (int i = 0; i < total_output; i++) {
        int input_idx[4];
        for (int d = 0; d < input_dims; d++) {
            input_idx[perm[d]] = idx[d];
        }

        int input_pos = get_position(input_idx, input_strides, input_dims);
        int output_pos = get_position(idx, output_strides, input_dims);

        output[output_pos] = input[input_pos];

        for (int d = input_dims - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < output_shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}
