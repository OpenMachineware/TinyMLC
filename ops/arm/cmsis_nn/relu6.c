/**
 * CMSIS-NN accelerated ReLU6 operator
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"


void tmlc_relu6_s8(const int8_t* input, int8_t* output, int size,
                   int32_t zero_point, int32_t input_scale_q,
                   int32_t output_scale_q)
{
    static int8_t temp_buf[16384];

    if (size > 16384) {
        // Fallback: pure integer version
        int32_t six_q15 = 196608;  // 6.0 in Q15

        for (int i = 0; i < size; i++) {
            int32_t x = ((int32_t)input[i] - zero_point) * input_scale_q;
            if (x < 0) x = 0;
            if (x > six_q15) x = six_q15;
            int32_t out = (x * 32768) / output_scale_q;
            out += zero_point;
            if (out > 127) out = 127;
            if (out < -128) out = -128;
            output[i] = (int8_t)out;
        }
        return;
    }

    for (int i = 0; i < size; i++) {
        temp_buf[i] = input[i];
    }

    arm_relu6_s8(temp_buf, size);

    for (int i = 0; i < size; i++) {
        output[i] = temp_buf[i];
    }
}
