#include "tinymlc.h"


void tmlc_hard_sigmoid_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int32_t zero_point,
    float input_scale,
    float output_scale)
{
    // Convert 3.0 and 6.0 to int8 scale
    float three_f = 3.0f / input_scale;
    float six_f = 6.0f / input_scale;
    int32_t three_q = (int32_t)(three_f + 0.5f);
    int32_t six_q = (int32_t)(six_f + 0.5f);

    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - zero_point;

        // Hard sigmoid: min(max(x + 3, 0), 6)
        int32_t y = x + three_q;
        if (y < 0) y = 0;
        if (y > six_q) y = six_q;

        // Divide by 6 (scale to output)
        float y_f = (float)y * input_scale / 6.0f;
        int32_t out = (int32_t)(y_f / output_scale + 0.5f);
        out += zero_point;

        if (out > 127) out = 127;
        if (out < -128) out = -128;
        output[i] = (int8_t)out;
    }
}
