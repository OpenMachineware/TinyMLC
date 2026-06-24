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

void tmlc_upsample_nearest_s8(const int8_t* input, int8_t* output,
                              int H, int W, int C,
                              int scale_h, int scale_w,
                              int32_t input_zero_point,
                              int32_t output_zero_point,
                              // scale * 2^15, Q15 format
                              int32_t input_scale_q,
                              // scale * 2^15, Q15 format
                              int32_t output_scale_q);

void tmlc_conv_transpose_s8(const int8_t* input, const int8_t* weights,
                            const int32_t* bias, int8_t* output,
                            int in_h, int in_w, int in_c,
                            int out_h, int out_w, int out_c,
                            int kernel_h, int kernel_w,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int32_t multiplier, int32_t shift,
                            int32_t input_zero_point,
                            int32_t output_zero_point);

void tmlc_global_avg_pool_s8(const int8_t* input, int8_t* output,
                             int H, int W, int C,
                             int32_t input_zero_point,
                             int32_t output_zero_point,
                             int32_t input_scale_q, int32_t output_scale_q);

void tmlc_flatten_s8(const int8_t* input, int8_t* output,
                     int in_h, int in_w, int in_c);

void tmlc_split_s8(const int8_t* input, int8_t** outputs, int H, int W, int C,
                   int* split_sizes, int num_splits, int axis);

void tmlc_argmax_s8(const int8_t* input, int32_t* output, int H, int W, int C,
                    int axis);

void tmlc_strided_slice_s8(const int8_t* input, int8_t* output,
                           int in_h, int in_w, int in_c,
                           int start_h, int start_w, int start_c,
                           int size_h, int size_w, int size_c,
                           int stride_h, int stride_w, int stride_c);

typedef struct {
    int16_t x1, y1, x2, y2;  // Fixed-point coordinates (Q7: 1/128 precision)
    int16_t score;           // Fixed-point score (Q7: 1/128 precision)
    int class_id;
    int keep;
} Box;

int tmlc_nms(Box* boxes, int num_boxes, int iou_threshold_q7,
             int max_output_size);

// y = (x > 0) ? x : x * alpha
// alpha is fixed-point Q7 format (alpha_q7 = alpha * 128)
// For alpha = 0.1, alpha_q7 = 13 (0.1 * 128 = 12.8 rounded)
void tmlc_leaky_relu_s8(const int8_t* input, int8_t* output, int size,
                        int16_t alpha_q7, int32_t zero_point);

// y = min(max(x, 0), 6)
void tmlc_relu6_s8(const int8_t* input, int8_t* output, int size,
                   int32_t zero_point,
                   int32_t input_scale_q,    // Q15 format
                   int32_t output_scale_q);  // Q15 format

// y = min(max(x + 3, 0), 6) / 6
void tmlc_hard_sigmoid_s8(const int8_t* input, int8_t* output, int size,
                          int32_t zero_point,
                          int32_t input_scale_q,    // Q15 format
                          int32_t output_scale_q);  // Q15 format

// y = (x > 0) ? x : x * alpha
// alpha is per-channel
void tmlc_prelu_s8(const int8_t* input, const int8_t* alpha, int8_t* output,
                   int H, int W, int C, int32_t zero_point);

// y = min(max(x, min_val), max_val)
void tmlc_clip_s8(const int8_t* input, int8_t* output, int size,
                  int8_t min_val, int8_t max_val, int32_t zero_point);

void tmlc_reduce_sum_s8(const int8_t* input, int32_t* output,
                        int H, int W, int C, int axis, int32_t zero_point);

#endif
