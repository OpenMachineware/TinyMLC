#include "tinymlc.h"
#include <stddef.h>

void tmlc_pad_s8(const int8_t* input,
                 const int32_t* paddings,
                 int8_t* output,
                 int input_dims,
                 const int* input_shape,
                 const int* output_shape)
{
    // 简单实现：只支持 4 维 padding
    // paddings 格式: [top, bottom, left, right, front, back, ...]
    // 对于 4 维 NHWC: [0, 0, top, bottom, left, right, 0, 0]

    if (input_dims != 4 || input == NULL || output == NULL || paddings == NULL) {
        return;
    }

    int pad_top = paddings[2];
    int pad_bottom = paddings[3];
    int pad_left = paddings[4];
    int pad_right = paddings[5];

    int input_h = input_shape[1];
    int input_w = input_shape[2];
    int input_c = input_shape[3];

    int output_h = output_shape[1];
    int output_w = output_shape[2];
    int output_c = output_shape[3];

    // 初始化输出为 0
    int output_size = output_h * output_w * output_c;
    for (int i = 0; i < output_size; i++) {
        output[i] = 0;
    }

    // 复制输入到 padding 后的位置
    for (int h = 0; h < input_h; h++) {
        for (int w = 0; w < input_w; w++) {
            for (int c = 0; c < input_c; c++) {
                int oh = h + pad_top;
                int ow = w + pad_left;
                output[oh * output_w * output_c + ow * output_c + c] =
                    input[h * input_w * input_c + w * input_c + c];
            }
        }
    }
}
