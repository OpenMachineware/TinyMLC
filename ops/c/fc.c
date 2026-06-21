#include "tinymlc.h"

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size,
                              int32_t multiplier,
                              int32_t shift)
{
    for (int out = 0; out < output_size; out++) {
        int32_t sum = bias ? bias[out] : 0;
        for (int in = 0; in < input_size; in++) {
            sum += (int32_t)input[in] * (int32_t)weights[out * input_size + in];
        }
        // int8 rescale: output = round((sum * multiplier) / 2^(31+shift))
        int64_t scaled = ((int64_t)sum * multiplier);
        scaled += (scaled >= 0) ? (1LL << 30) : -(1LL << 30);
        scaled >>= (31 + shift);
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        output[out] = (int8_t)scaled;
    }
}
