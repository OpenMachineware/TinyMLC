#include "tinymlc.h"


void tmlc_prelu_s8(
    const int8_t* input,
    const int8_t* alpha,
    int8_t* output,
    int H, int W, int C,
    int32_t zero_point)
{
    int total = H * W * C;

    for (int i = 0; i < total; i++) {
        int c = i % C;
        int32_t x = (int32_t)input[i] - zero_point;
        int32_t a = (int32_t)alpha[c];

        int32_t y;
        if (x > 0) {
            y = x;
        } else {
            // x * alpha (both are int8, product is int16)
            y = (x * a) / 128;  // assuming alpha is Q7 format
        }

        y += zero_point;
        if (y > 127) y = 127;
        if (y < -128) y = -128;
        output[i] = (int8_t)y;
    }
}
