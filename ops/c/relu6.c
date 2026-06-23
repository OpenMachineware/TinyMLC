#include "tinymlc.h"


void tmlc_relu6_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int32_t zero_point,
    float input_scale,
    float output_scale)
{
    // Convert 6.0 to int8 scale
    float six_f = 6.0f / input_scale;
    int32_t six_q = (int32_t)(six_f + 0.5f);

    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - zero_point;

        // ReLU6: clamp to [0, 6]
        if (x < 0) x = 0;
        if (x > six_q) x = six_q;

        // Dequantize to output scale
        float x_f = (float)x * input_scale;
        int32_t out = (int32_t)(x_f / output_scale + 0.5f);
        out += zero_point;

        if (out > 127) out = 127;
        if (out < -128) out = -128;
        output[i] = (int8_t)out;
    }
}
