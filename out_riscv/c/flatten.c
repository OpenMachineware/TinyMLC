#include "tinymlc.h"


void tmlc_flatten_s8(
    const int8_t* input,
    int8_t* output,
    int in_h, int in_w, int in_c)
{
    int total = in_h * in_w * in_c;
    for (int i = 0; i < total; i++) {
        output[i] = input[i];
    }
}
