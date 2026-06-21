#include "tinymlc.h"

void tmlc_concat_s8(const int8_t** inputs, const int* sizes, int num_inputs,
                    int8_t* output)
{
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        // Manual copy, avoid using memcpy
        for (int j = 0; j < sizes[i]; j++) {
            output[offset + j] = inputs[i][j];
        }
        offset += sizes[i];
    }
}