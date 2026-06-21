#include <stddef.h>
#include "tinymlc.h"

// Calculate position in tensor
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

    // Calculate output shape
    int output_shape[4];
    for (int i = 0; i < input_dims; i++) {
        output_shape[i] = input_shape[perm[i]];
    }

    // Calculate input strides
    int input_strides[4];
    input_strides[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; i--) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Calculate output strides
    int output_strides[4];
    output_strides[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; i--) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Iterate over all output positions
    int idx[4] = {0, 0, 0, 0};
    int total_output = 1;
    for (int i = 0; i < input_dims; i++) {
        total_output *= output_shape[i];
    }

    for (int i = 0; i < total_output; i++) {
        // Calculate input indices
        int input_idx[4];
        for (int d = 0; d < input_dims; d++) {
            input_idx[perm[d]] = idx[d];
        }

        int input_pos = get_position(input_idx, input_strides, input_dims);
        int output_pos = get_position(idx, output_strides, input_dims);

        output[output_pos] = input[input_pos];

        // Update indices
        for (int d = input_dims - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < output_shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}

/*
void tmlc_transpose_s8(const int8_t* input,
                       const int32_t* perm,
                       int8_t* output,
                       int input_dims,
                       const int* input_shape)
{
    if (input_dims <= 0 || input == NULL || output == NULL) {
        return;
    }

    // Calculate output shape
    int output_shape[4];
    for (int i = 0; i < input_dims; i++) {
        output_shape[i] = input_shape[perm[i]];
    }

    // Calculate total element count
    int total_size = 1;
    for (int i = 0; i < input_dims; i++) {
        total_size *= input_shape[i];
    }

    // Calculate input and output strides
    int input_strides[4];
    int output_strides[4];

    input_strides[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; i--) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    output_strides[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; i--) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Recursive transpose
    int input_idx[4] = {0, 0, 0, 0};
    int output_idx[4] = {0, 0, 0, 0};

    for (int i = 0; i < total_size; i++) {
        // Calculate input position
        int input_pos = 0;
        for (int d = 0; d < input_dims; d++) {
            input_pos += input_idx[d] * input_strides[d];
        }

        // Calculate output position
        int output_pos = 0;
        for (int d = 0; d < input_dims; d++) {
            output_pos += output_idx[d] * output_strides[d];
        }

        output[output_pos] = input[input_pos];

        // Update indices (in input order)
        for (int d = input_dims - 1; d >= 0; d--) {
            input_idx[d]++;
            if (input_idx[d] < input_shape[d]) {
                break;
            }
            input_idx[d] = 0;
            // Update output indices
            int out_d = perm[d];
            output_idx[out_d]++;
            if (output_idx[out_d] < output_shape[out_d]) {
                break;
            }
            output_idx[out_d] = 0;
        }
    }
}
*/
