// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

#include <stdint.h>
#include "tinymlc.h"

#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

// 权重（占位，后续从 tflite 提取）
static const int8_t fc_weights[5600] = {0};
static const int32_t fc_bias[10] = {0};

// 推理函数
void run_inference(const int8_t* input, int8_t* output) {
    int8_t fc_input[560];
    int8_t fc_output[10];

    // 占位：简单复制输入到 fc_input
    for (int i = 0; i < 560; i++) {
        fc_input[i] = input[i % INPUT_SIZE];
    }

    tmlc_fully_connected_s8(fc_input, fc_weights, fc_bias, fc_output, 560, 10);
    tmlc_softmax_s8(fc_output, output, 10);
}

// 主函数
int main() {
    static int8_t input[INPUT_SIZE];
    static int8_t output[OUTPUT_SIZE];

    // 测试输入
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1;
    }

    run_inference(input, output);

    // 输出结果（占位，后续替换为实际输出）
    // 暂时什么也不做，避免编译问题

    return 0;
}
