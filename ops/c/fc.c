#include "tinymlc.h"

/**
 * int8 quantized fully connected operator
 *
 * Quantization formula:
 *   output = round((acc * multiplier) >> (31 + shift))
 *   where acc = sum(input * weight) + bias
 *
 * multiplier and shift are calculated by compiler based on quantization params:
 *   effective_scale = (input_scale * weight_scale) / output_scale
 *   multiplier = effective_scale * 2^31 (adjusted to Q31 range)
 *   shift adjusts multiplier to valid range
 *
 * Value range:
 *   127 / -128: int8 symmetric quantization range
 *   1 << 30: round-to-nearest offset (0.5 * 2^31)
 *   31: Q31 fixed-point format
 */
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
        // rescale: output = round((sum * multiplier) / 2^(31+shift))
        int64_t scaled = ((int64_t)sum * multiplier);
        scaled += (scaled >= 0) ? (1LL << 30) : -(1LL << 30);  // round-to-nearest
        scaled >>= (31 + shift);
        if (scaled > 127) scaled = 127;   // int8 max
        if (scaled < -128) scaled = -128; // int8 min
        output[out] = (int8_t)scaled;
    }
}
