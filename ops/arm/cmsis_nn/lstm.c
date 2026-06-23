/**
 * CMSIS-NN accelerated LSTM operator
 *
 * Note: CMSIS-NN 7.0.0 LSTM API is complex with separate gate parameters.
 * This implementation falls back to pure C for actual computation.
 *
 * Wraps CMSIS-NN LSTM functions to match TinyMLC's tmlc_unidirectional_sequence_lstm_s8
 * interface.
 */

#include "tinymlc.h"
#include "arm_nnfunctions.h"

// Pure C LSTM implementation (always used - CMSIS-NN LSTM API is too complex)
extern void tmlc_unidirectional_sequence_lstm_s8_c(
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
    int hidden_size)
{
    // Fall back to pure C implementation
    // CMSIS-NN 7.0.0 LSTM API requires complex gate parameter setup
    tmlc_unidirectional_sequence_lstm_s8_c(input, input_weights, recurrent_weights,
                                           bias, output_sequence, output_state,
                                           cell_state, time_steps, batch_size,
                                           input_size, hidden_size);
}
