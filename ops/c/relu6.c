#include "tinymlc.h"


void tmlc_relu6_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    int32_t zero_point,
    int32_t input_scale_q,   // Q15 format
    int32_t output_scale_q)  // Q15 format
{
    // 6.0 in Q15: 6 * 32768 = 196608
    int32_t six_q15 = 196608;

    for (int i = 0; i < size; i++) {
        // Dequantize input to Q15
        int32_t x = ((int32_t)input[i] - zero_point) * input_scale_q;

        // ReLU6: clamp to [0, 6]
        if (x < 0) x = 0;
        if (x > six_q15) x = six_q15;

        // Quantize to output scale
        int32_t out = (x * 32768) / output_scale_q;
        out += zero_point;

        if (out > 127) out = 127;
        if (out < -128) out = -128;

        output[i] = (int8_t)out;
    }
}
