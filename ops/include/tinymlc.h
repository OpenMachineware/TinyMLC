#ifndef TINYMLC_H
#define TINYMLC_H

#include <stdint.h>


typedef int8_t  s8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef int32_t s32;
typedef uint32_t u32;

void tmlc_fully_connected_s8(const s8* input,
                              const s8* weights,
                              const s32* bias,
                              s8* output,
                              int input_size,
                              int output_size,
                              int32_t multiplier,
                              int32_t shift);

void tmlc_softmax_s8(const s8* input, s8* output, int size);

void tmlc_unidirectional_sequence_lstm_s8(
    const int8_t* input,
    const int8_t* input_weights,
    const int8_t* recurrent_weights,
    const int32_t* bias,
    int8_t* output_sequence,
    int8_t* output_state,
    int8_t* cell_state,
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size);

void tmlc_reshape_s8(const int8_t* input, int8_t* output,
                     int input_size, const int* new_shape, int shape_size);

void tmlc_add_s8(const int8_t* input1, const int8_t* input2,
                 int8_t* output, int size);

void tmlc_svdf_s8(const int8_t* input, const int8_t* weights,
                  const int32_t* bias, int8_t* output, int time_steps,
                  int input_size, int rank, int units);

void tmlc_conv2d_s8(const int8_t* input,
                    const int8_t* weights,
                    const int32_t* bias,
                    int8_t* output,
                    int input_h, int input_w, int input_c,
                    int output_h, int output_w, int output_c,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int padding_h, int padding_w,
                    int32_t multiplier, int32_t shift);

void tmlc_max_pool_2d_s8(const int8_t* input,
                         int8_t* output,
                         int input_h, int input_w, int input_c,
                         int output_h, int output_w, int output_c,
                         int pool_h, int pool_w,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w);

void tmlc_depthwise_conv_2d_s8(const int8_t* input,
                               const int8_t* weights,
                               const int32_t* bias,
                               int8_t* output,
                               int input_h, int input_w, int input_c,
                               int output_h, int output_w, int output_c,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int depth_multiplier,
                               int padding_h, int padding_w,
                               int32_t multiplier, int32_t shift);

void tmlc_relu_s8(const int8_t* input, int8_t* output, int size);

void tmlc_avg_pool_2d_s8(const int8_t* input,
                         int8_t* output,
                         int input_h, int input_w, int input_c,
                         int output_h, int output_w, int output_c,
                         int pool_h, int pool_w,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w);

void tmlc_transpose_s8(const int8_t* input,
                       const int32_t* perm,
                       int8_t* output,
                       int input_dims,
                       const int* input_shape);

void tmlc_pad_s8(const int8_t* input,
                 const int32_t* paddings,
                 int8_t* output,
                 int input_dims,
                 const int* input_shape,
                 const int* output_shape);

void tmlc_mean_s8(const int8_t* input,
                  int8_t* output,
                  int input_dims,
                  const int* input_shape,
                  const int* output_shape,
                  const int32_t* axis,
                  int axis_count,
                  int keep_dims);

void tmlc_multiply_s8(const int8_t* input1, const int8_t* input2,
                      int8_t* output, int size);

void tmlc_sigmoid_s8(const int8_t* input, int8_t* output, int size);

void tmlc_concat_s8(const int8_t** inputs, const int* sizes, int num_inputs,
                    int8_t* output);

void tmlc_sub_s8(const int8_t* input1, const int8_t* input2,
                 int8_t* output, int size);

void tmlc_tanh_s8(const int8_t* input, int8_t* output, int size);


#endif
