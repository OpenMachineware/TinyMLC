/**
 * NMSIS-NN accelerated convolution operator
 *
 * Wraps riscv_convolve_wrapper_s8 to match TinyMLC's tmlc_conv2d_s8 interface.
 */

#include "tinymlc.h"
#include "riscv_nnfunctions.h"

static nmsis_nn_context ctx;
static int8_t nmsis_nn_buf[16384];

void tmlc_conv2d_s8(const int8_t* input,
                    const int8_t* weights,
                    const int32_t* bias,
                    int8_t* output,
                    int input_h, int input_w, int input_c,
                    int output_h, int output_w, int output_c,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int padding_h, int padding_w,
                    int32_t multiplier, int32_t shift)
{
    nmsis_nn_conv_params conv_params = {
        .input_offset = 0,
        .output_offset = 0,
        .stride = {.h = stride_h, .w = stride_w},
        .padding = {.h = padding_h, .w = padding_w},
        .dilation = {.h = 1, .w = 1},
        .activation = {.min = -128, .max = 127}
    };

    // Per-channel quantization (NMSIS-NN uses pointers for multiplier/shift)
    static int32_t multiplier_arr[1];
    static int32_t shift_arr[1];
    multiplier_arr[0] = multiplier;
    shift_arr[0] = shift;

    nmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multiplier_arr,
        .shift = shift_arr
    };

    nmsis_nn_dims input_dims = {
        .n = 1, .h = input_h, .w = input_w, .c = input_c};
    nmsis_nn_dims filter_dims = {
        .n = 1, .h = kernel_h, .w = kernel_w, .c = input_c};
    nmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = output_c};
    nmsis_nn_dims output_dims = {
        .n = 1, .h = output_h, .w = output_w, .c = output_c};

    ctx.buf = nmsis_nn_buf;
    ctx.size = sizeof(nmsis_nn_buf);

    riscv_convolve_wrapper_s8(&ctx, &conv_params, &quant_params,
                              &input_dims, input,
                              &filter_dims, weights,
                              &bias_dims, bias,
                              &output_dims, output);
}