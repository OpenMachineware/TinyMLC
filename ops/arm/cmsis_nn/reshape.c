/**
 * CMSIS-NN accelerated reshape operator
 *
 * Wraps arm_reshape_s8 to match TinyMLC's tmlc_reshape_s8 interface.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_reshape_s8(const int8_t* input, int8_t* output,
                     int input_size, const int* new_shape, int shape_size)
{
    // Calculate total size from new_shape
    int output_size = 1;
    for (int i = 0; i < shape_size; i++) {
        output_size *= new_shape[i];
    }

    if (input_size != output_size) return;

    // Call CMSIS-NN reshape
    arm_reshape_s8(input, output, input_size);
}
