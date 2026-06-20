#include "tinymlc.h"
#include "arm_nnfunctions.h"
#include "model.h"
#include "debug_print.h"

void tmlc_fully_connected_s8(const int8_t* input,
                              const int8_t* weights,
                              const int32_t* bias,
                              int8_t* output,
                              int input_size,
                              int output_size)
{
    // 1. 缓冲区
    cmsis_nn_dims filter_dims;
    filter_dims.n = output_size;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = input_size;

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    static int8_t fc_buffer[4096];

    cmsis_nn_context ctx;
    ctx.buf = fc_buffer;
    ctx.size = buf_size;

    // 2. FC 参数
    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = -128;
    fc_params.activation.max = 127;

    // 3. 量化参数
    cmsis_nn_per_tensor_quant_params quant_params;
    quant_params.multiplier = 1610612736;
    quant_params.shift = -7;

    // 4. 输入维度
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = input_size;

    // 5. 输出维度
    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = output_size;

    // 6. bias 维度
    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_size;

DEBUG_STR("转置\n");
int8_t weights_transposed[input_size * output_size];
for (int out = 0; out < output_size; out++) {
    for (int in = 0; in < input_size; in++) {
        weights_transposed[in * output_size + out] = weights[out * input_size + in];
    }
}

DEBUG_STR("FC: pure C output[0..9]=");
for (int out = 0; out < 10 && out < output_size; out++) {
    int32_t sum = bias ? bias[out] : 0;
    for (int in = 0; in < input_size; in++) {
        sum += (int32_t)input[in] * (int32_t)weights_transposed[out * input_size + in];
    }
    int8_t pure_out = (int8_t)(sum >> 8);  // 和之前纯 C 版本一样的量化
    DEBUG_INT(pure_out);
    DEBUG_CHAR(' ');
}
DEBUG_ENDL();


DEBUG_STR("FC: input_offset=");
DEBUG_INT(fc_params.input_offset);
DEBUG_ENDL();

DEBUG_STR("FC: output_offset=");
DEBUG_INT(fc_params.output_offset);
DEBUG_ENDL();

DEBUG_STR("FC: multiplier=");
DEBUG_INT(quant_params.multiplier);
DEBUG_ENDL();

DEBUG_STR("FC: shift=");
DEBUG_INT(quant_params.shift);
DEBUG_ENDL();

DEBUG_STR("FC: CMSIS-NN input[0..9]=");
for (int i = 0; i < 10 && i < input_size; i++) {
    DEBUG_INT(input[i]);
    DEBUG_CHAR(' ');
}
DEBUG_ENDL();

int8_t scaled_input[input_size];
for (int i = 0; i < input_size; i++) {
    scaled_input[i] = (int8_t)((int32_t)input[i] + 128);
}

    arm_cmsis_nn_status status = arm_fully_connected_s8(
        &ctx,
        &fc_params,
        &quant_params,
        &input_dims,
        scaled_input,
        &filter_dims,
        weights_transposed,
        &bias_dims,
        bias,
        &output_dims,
        output
    );
DEBUG_STR("FC: CMSIS-NN shifted output[0..9]=");
for (int i = 0; i < 10 && i < output_size; i++) {
    int8_t shifted = output[i] >> 1;
    DEBUG_INT(shifted);
    DEBUG_CHAR(' ');
}
DEBUG_ENDL();

for (int i = 0; i < output_size; i++) {
    output[i] = (int8_t)((int32_t)output[i] + 3);
}
}
