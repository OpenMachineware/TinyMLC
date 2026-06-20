#include "tinymlc.h"

void tmlc_reshape_s8(const int8_t* input, int8_t* output,
                     int input_size, const int* new_shape, int shape_size)
{
    int output_size = 1;
    for (int i = 0; i < shape_size; i++) {
        output_size *= new_shape[i];
    }

    if (input_size != output_size) return;

    for (int i = 0; i < input_size; i++) {
        output[i] = input[i];
    }
}
