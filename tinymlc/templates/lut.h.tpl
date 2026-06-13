// 自动生成的 LUT 表，请勿手动修改
// sigmoid 和 tanh 的 int16 查找表（256 条目）

#ifndef TINYMLC_LUT_H
#define TINYMLC_LUT_H

#include <stdint.h>

// Sigmoid LUT（输入范围 [-8,8)，输出范围 [0,1] 量化到 [0,32767]）
static const int16_t sigmoid_lut[256] = {
{% for val in sigmoid_lut %}
    {{ val }}{% if not loop.last %},{% endif %}
{% endfor %}
};

// Tanh LUT（输入范围 [-8,8)，输出范围 [-1,1] 量化到 [-32768,32767]）
static const int16_t tanh_lut[256] = {
{% for val in tanh_lut %}
    {{ val }}{% if not loop.last %},{% endif %}
{% endfor %}
};

// LUT 查找函数（线性插值）
static inline int16_t sigmoid_lut_lookup(int32_t x) {
    // x 是 int32 累加值，需要根据量化参数映射到 [-8,8) 范围
    // 假设 x 已经缩放到 [0, 256*8) 范围
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
