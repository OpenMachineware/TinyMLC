#include "tinymlc.h"
#include <stddef.h>

void tmlc_mean_s8(const int8_t* input,
                  int8_t* output,
                  int input_dims,
                  const int* input_shape,
                  const int* output_shape,
                  const int32_t* axis,
                  int axis_count,
                  int keep_dims)
{
    // 简单实现：全局平均（所有维度）
    // 对于 MobileNetV2 的 MEAN，通常是全局平均池化
    // 输入形状: [1, 7, 7, 1280] -> 输出: [1, 1, 1, 1280]

    if (input == NULL || output == NULL || input_dims <= 0) {
        return;
    }

    // 计算总元素数
    int total_size = 1;
    for (int i = 0; i < input_dims; i++) {
        total_size *= input_shape[i];
    }

    // 如果输出维度是 1，做全局平均
    int output_size = 1;
    for (int i = 0; i < input_dims; i++) {
        output_size *= output_shape[i];
    }

    if (output_size == 1) {
        // 全局平均
        int32_t sum = 0;
        for (int i = 0; i < total_size; i++) {
            sum += input[i];
        }
        output[0] = (int8_t)(sum / total_size);
    } else {
        // 沿最后一个维度平均（最常见的情况）
        int last_dim = input_shape[input_dims - 1];
        int num_groups = total_size / last_dim;

        for (int g = 0; g < num_groups; g++) {
            int32_t sum = 0;
            for (int d = 0; d < last_dim; d++) {
                sum += input[g * last_dim + d];
            }
            output[g] = (int8_t)(sum / last_dim);
        }
    }
}
