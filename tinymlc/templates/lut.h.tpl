// Auto-generated LUT tables, do not modify manually
// sigmoid and tanh int16 lookup tables (256 entries)

#ifndef TINYMLC_LUT_H
#define TINYMLC_LUT_H

#include <stdint.h>

// Sigmoid LUT (input range [-8,8), output range [0,1] quantized to [0,32767])
static const int16_t sigmoid_lut[256] = {
{% for val in sigmoid_lut %}
    {{ val }}{% if not loop.last %},{% endif %}
{% endfor %}
};

// Tanh LUT (input range [-8,8), output range [-1,1] quantized to [-32768,32767])
static const int16_t tanh_lut[256] = {
{% for val in tanh_lut %}
    {{ val }}{% if not loop.last %},{% endif %}
{% endfor %}
};

// LUT lookup functions (linear interpolation)
static inline int16_t sigmoid_lut_lookup(int32_t x) {
    // x is int32 accumulator, needs to be mapped to [-8,8) range based on quantization params
    // Assume x is already scaled to [0, 256*8) range
    int32_t idx = (x >> 8) & 0xFF;
    uint8_t frac = (uint8_t)(x & 0xFF);

    int16_t lower = sigmoid_lut[idx];
    int16_t upper = sigmoid_lut[idx + 1];

    int32_t result = lower + ((upper - lower) * frac >> 8);
    return (int16_t)result;
}

static inline int16_t tanh_lut_lookup(int32_t x) {
    int32_t idx = (x >> 8) & 0xFF;
    uint8_t frac = (uint8_t)(x & 0xFF);

    int16_t lower = tanh_lut[idx];
    int16_t upper = tanh_lut[idx + 1];

    int32_t result = lower + ((upper - lower) * frac >> 8);
    return (int16_t)result;
}

#endif // TINYMLC_LUT_H
