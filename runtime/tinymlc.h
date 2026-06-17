#ifndef TINYMLC_H
#define TINYMLC_H

#include <stdint.h>

// 数据类型别名
typedef int8_t  s8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef int32_t s32;
typedef uint32_t u32;

// FC
void tmlc_fully_connected_s8(const s8* input,
                              const s8* weights,
                              const s32* bias,
                              s8* output,
                              int input_size,
                              int output_size);

// Softmax
void tmlc_softmax_s8(const s8* input, s8* output, int size);

// LSTM
void tmlc_unidirectional_sequence_lstm_s8(
    const int8_t* input,
    const int8_t* input_weights,
    const int8_t* recurrent_weights,
    const int32_t* bias,
    int8_t* output_sequence,   // 完整序列 [time_steps, hidden_size]
    int8_t* output_state,      // 最后状态 [hidden_size]（可 NULL）
    int8_t* cell_state,        // 最后细胞状态（可 NULL）
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size);

// Reshape
void tmlc_reshape_s8(const int8_t* input, int8_t* output,
                     int input_size, const int* new_shape, int shape_size);

// ADD 算子：逐元素相加
void tmlc_add_s8(const int8_t* input1, const int8_t* input2, int8_t* output, int size);

// SVDF 算子：用于关键词识别等序列任务
void tmlc_svdf_s8(const int8_t* input, const int8_t* weights,
                  const int32_t* bias, int8_t* output, int time_steps,
                  int input_size, int rank, int units);

#endif
