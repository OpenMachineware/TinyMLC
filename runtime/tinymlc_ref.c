#include "tinymlc.h"


void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size) {
    for (int out = 0; out < output_size; out++) {
        int32_t sum = bias ? bias[out] : 0;
        for (int in = 0; in < input_size; in++) {
            sum += (int32_t)input[in] * (int32_t)weights[out * input_size + in];
        }
        // FIXME 简单的量化缩放（后续可以优化）
        output[out] = (int8_t)(sum >> 8);
    }
}

void tmlc_softmax_s8(const s8* input, s8* output, int size) {
    // 找最大值
    s8 max = -128;
    for (int i = 0; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    // 计算 exp 近似值（整数版本）
    s32 sum = 0;
    s32 exp_vals[10];
    for (int i = 0; i < size; i++) {
        s32 x = (s32)(input[i] - max);
        s32 exp_val = (x >= 0) ? (1 << (x / 4)) : (1 >> ((-x) / 4));
        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    // 归一化
    for (int i = 0; i < size; i++) {
        s32 prob = (exp_vals[i] * 128) / sum;
        output[i] = (s8)prob;
    }
}
