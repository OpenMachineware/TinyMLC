#include "tinymlc.h"

void tmlc_add_s8(const int8_t* input1, const int8_t* input2,
                  int8_t* output, int size)
{
    for (int i = 0; i < size; i++) {
        int32_t sum = (int32_t)input1[i] + (int32_t)input2[i];
        output[i] = (int8_t)(sum >> 1);
    }
}
