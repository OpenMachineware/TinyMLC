#include "tinymlc.h"
#include "arm_nnfunctions.h"
#include "model.h"

void tmlc_conv2d_s8(const int8_t* input,
                    const int8_t* weights,
                    const int32_t* bias,
                    int8_t* output,
                    int input_h, int input_w, int input_c,
                    int output_h, int output_w, int output_c,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int padding_h, int padding_w)
{
    // 1. 缓冲区
    cmsis_nn_dims input_dims_buf;
    input_dims_buf.n = 1;
    input_dims_buf.h = input_h;
    input_dims_buf.w = input_w;
    input_dims_buf.c = input_c;

    cmsis_nn_dims filter_dims_buf;
    filter_dims_buf.n = output_c;
    filter_dims_buf.h = kernel_h;
    filter_dims_buf.w = kernel_w;
    filter_dims_buf.c = input_c;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims_buf, &filter_dims_buf);
    static int8_t conv_buffer[8192];

    cmsis_nn_context ctx;
    ctx.buf = conv_buffer;
    ctx.size = buf_size;

    // 2. Conv 参数
    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.stride.h = stride_h;
    conv_params.stride.w = stride_w;
    conv_params.padding.h = padding_h;
    conv_params.padding.w = padding_w;
    conv_params.dilation.h = 1;
    conv_params.dilation.w = 1;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    // 3. 量化参数（per_channel）
    int32_t multipliers[output_c];
    int32_t shifts[output_c];
    for (int i = 0; i < output_c; i++) {
        multipliers[i] = 410762;
        shifts[i] = -32;
    }
    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = multipliers;
    quant_params.shift = shifts;

    // 4. 输入维度
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = input_h;
    input_dims.w = input_w;
    input_dims.c = input_c;

    // 5. 输出维度
    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = output_h;
    output_dims.w = output_w;
    output_dims.c = output_c;

    // 6. bias 维度
    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_c;

    // 7. 权重维度
    cmsis_nn_dims filter_dims;
    filter_dims.n = output_c;
    filter_dims.h = kernel_h;
    filter_dims.w = kernel_w;
    filter_dims.c = input_c;

    arm_cmsis_nn_status status = arm_convolve_s8(
        &ctx,
        &conv_params,
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
