#include "tinymlc.h"

// void tmlc_fully_connected_s8(const s8* input,
//                               const s8* weights,
//                               const s32* bias,
//                               s8* output,
//                               int input_size,
//                               int output_size) {
//     for (int out = 0; out < output_size; out++) {
//         s32 sum = bias ? bias[out] : 0;
//         for (int in = 0; in < input_size; in++) {
//             sum += (s32)input[in] * (s32)weights[out * input_size + in];
//         }
//         // 简单的量化缩放（后续可以优化）
//         output[out] = (s8)(sum >> 8);
//     }
// }

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size) {
    volatile char* uart = (volatile char*)0x10000000;
//     uart[0] = '1'; uart[0] = '\n';

    output[0] = 42;
//     uart[0] = '2'; uart[0] = '\n';

    // 如果这里有问题，加更多打印
    for (int i = 0; i < output_size; i++) {
        output[i] = 42 - i;
    }
//     uart[0] = '3'; uart[0] = '\n';
}

void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size) {
    volatile char* uart = (volatile char*)0x10000000;
//     uart[0] = 'S'; uart[0] = '1'; uart[0] = '\n';

    // 找最大值
    int8_t max = -128;
    for (int i = 0; i < size; i++) {
        if (input[i] > max) max = input[i];
    }
//     uart[0] = 'S'; uart[0] = '2'; uart[0] = '\n';

    // 计算 exp 近似值
    int32_t sum = 0;
    int32_t exp_vals[10];
    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)(input[i] - max);
        int32_t exp_val = (x >= 0) ? (1 << (x / 4)) : (1 >> ((-x) / 4));
        exp_vals[i] = exp_val;
        sum += exp_val;
    }
//     uart[0] = 'S'; uart[0] = '3'; uart[0] = '\n';


    // 打印 sum 的值
//     uart[0] = 's'; uart[0] = '=';
    // 简单打印 sum 的低位字节
//     uart[0] = '0' + (sum & 0xF); uart[0] = '\n';

    // 归一化
//     uart[0] = 'L'; uart[0] = 'O'; uart[0] = 'O'; uart[0] = 'P'; uart[0] = '\n';
//     uart[0] = 's'; uart[0] = 'i'; uart[0] = 'z'; uart[0] = 'e'; uart[0] = '=';
//     uart[0] = '0' + (size / 10);
//     uart[0] = '0' + (size % 10);
//     uart[0] = '\n';
    for (int i = 0; i < size; i++) {
//         output[i] = input[i];
//         int32_t prob = (exp_vals[i] * 128) / sum;
//         output[i] = (int8_t)prob;
        // 打印当前 i
//         uart[0] = '0' + (i / 10);
//         uart[0] = '0' + (i % 10);
//         uart[0] = ':';
//         uart[0] = '\n';

        // 打印 input[i] 的值
        int8_t val = input[i];
//         uart[0] = '0' + (val / 10);
//         uart[0] = '0' + (val % 10);
//         uart[0] = '\n';

        output[i] = val;
        uart[0] = 'O'; uart[0] = 'K'; uart[0] = '\n';
    }
    uart[0] = 'S'; uart[0] = '4'; uart[0] = '\n';
}

// void tmlc_softmax_s8(const s8* input, s8* output, int size) {
//     // 找最大值
//     s8 max = -128;
//     for (int i = 0; i < size; i++) {
//         if (input[i] > max) max = input[i];
//     }
//
//     // 计算 exp 近似值（整数版本）
//     s32 sum = 0;
//     s32 exp_vals[10];
//     for (int i = 0; i < size; i++) {
//         s32 x = (s32)(input[i] - max);
//         s32 exp_val = (x >= 0) ? (1 << (x / 4)) : (1 >> ((-x) / 4));
//         exp_vals[i] = exp_val;
//         sum += exp_val;
//     }
//
//     // 归一化
//     for (int i = 0; i < size; i++) {
//         s32 prob = (exp_vals[i] * 128) / sum;
//         output[i] = (s8)prob;
//     }
// }
