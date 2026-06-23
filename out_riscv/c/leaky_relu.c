#include "tinymlc.h"


void tmlc_leaky_relu_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int16_t alpha_q7,
    int32_t zero_point)
{
    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - zero_point;
        int32_t y;

        if (x > 0) {
            y = x;
        } else {
            // x * alpha in Q7: (x * alpha_q7) / 128
            y = (x * alpha_q7) / 128;
        }

        y += zero_point;
        if (y > 127) y = 127;
        if (y < -128) y = -128;
        output[i] = (int8_t)y;
    }
}
