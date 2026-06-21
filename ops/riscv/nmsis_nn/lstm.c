/**
 * NMSIS-NN accelerated LSTM operator
 *
 * Wraps NMSIS-NN LSTM functions to match TinyMLC's tmlc_unidirectional_sequence_lstm_s8
 * interface.
 *
 * Note: NMSIS-NN requires specific buffer sizes for LSTM operations.
 * The scratch buffer is allocated internally.
 */

#include "tinymlc.h"
#include "rnn_nnfunctions.h"

// NMSIS-NN context for optimized operations (scratch buffer)
static cmsis_nn_context ctx;

// Buffer for NMSIS-NN LSTM internal operations
// Size depends on batch_size and hidden_size
static int8_t nmsis_nn_lstm_buf[16384];

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
    // NMSIS-NN LSTM parameters
    cmsis_nn_lstm_params lstm_params = {
        .input_offset = 0,
        .output_offset = 0,
        .hidden_state_offset = 0,
        .cell_state_offset = 0,
        .activation = {.min = -128, .max = 127},
    };

    cmsis_nn_per_tensor_quant_params quant_params = {
        .multiplier = 1,  // TODO: calculate from model scales
        .shift = 8,       // TODO: calculate from model scales
    };

    // NMSIS-NN dimensions
    cmsis_nn_dims input_dims = {
        .n = batch_size, .h = 1, .w = 1, .c = input_size
    };
    cmsis_nn_dims output_dims = {
        .n = batch_size, .h = 1, .w = 1, .c = hidden_size
    };

    // Input gate weights (i): input_weights[0 : hidden_size * input_size]
    // Forget gate weights (f): input_weights[hidden_size * input_size : 2 * hidden_size * input_size]
    // Cell gate weights (g): input_weights[2 * hidden_size * input_size : 3 * hidden_size * input_size]
    // Output gate weights (o): input_weights[3 * hidden_size * input_size : 4 * hidden_size * input_size]

    // Recurrent weights layout is the same
    // Bias layout: [bi, bf, bg, bo] each of size hidden_size

    ctx.buf = nmsis_nn_lstm_buf;
    ctx.size = sizeof(nmsis_nn_lstm_buf);

    // Call NMSIS-NN LSTM function when available
    // For now, call pure C implementation as fallback
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

    tmlc_unidirectional_sequence_lstm_s8_c(input, input_weights, recurrent_weights,
                                            bias, output_sequence, output_state,
                                            cell_state, time_steps, batch_size,
                                            input_size, hidden_size);
}