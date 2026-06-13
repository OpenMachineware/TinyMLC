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

// 全连接 (Fully Connected)
void tmlc_fully_connected_s8(const s8* input,
                              const s8* weights,
                              const s32* bias,
                              s8* output,
                              int input_size,
                              int output_size);

// Softmax
void tmlc_softmax_s8(const s8* input, s8* output, int size);

// LSTM (先放接口，实现可以稍后)
void tmlc_unidirectional_sequence_lstm_s8(
    const s8* input,                // [time_steps, batch, input_size]
    const s8* input_weights,        // [4, hidden, input_size]
    const s8* recurrent_weights,    // [4, hidden, hidden]
    const s32* bias,                // [4, hidden]
    s8* output_state,               // [batch, hidden]
    s8* cell_state,                 // [batch, hidden]
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size);

#endif
