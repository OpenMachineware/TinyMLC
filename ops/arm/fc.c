#include "tinymlc.h"
#include "arm_nnfunctions.h"

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size)
{
    // 1. 上下文（CMSIS-NN 7.0 需要）
    cmsis_nn_context ctx = {0};
    ctx.buf = NULL;
    ctx.size = 0;

    // 2. FC 参数
    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = -128;
    fc_params.activation.max = 127;

    // 3. 量化参数
    cmsis_nn_per_tensor_quant_params quant_params;
    quant_params.multiplier = 1073741824;  // 1.0 的 Q31 表示
    quant_params.shift = -6;

    // 4. 输入维度
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = input_size;

    // 5. 权重维度
    cmsis_nn_dims filter_dims;
    filter_dims.n = output_size;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = input_size;

    // 6. bias 维度
    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_size;

    // 7. 输出维度
    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = output_size;

    arm_fully_connected_s8(
        &ctx,
        &fc_params,
        &quant_params,
        &input_dims,
        input,
        &filter_dims,
        weights,
        &bias_dims,
        bias,
        &output_dims,
        output
    );
}
