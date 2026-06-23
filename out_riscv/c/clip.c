#include "tinymlc.h"


void tmlc_clip_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int8_t min_val,
    int8_t max_val,
    int32_t zero_point)
{
    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - zero_point;

        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;

        x += zero_point;
        if (x > 127) x = 127;
        if (x < -128) x = -128;
        output[i] = (int8_t)x;
    }
}
