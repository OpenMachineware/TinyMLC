/* TinyMLC - Tiny Machine Learning Compiler
*
 * Copyright (c) 2026 Jia Liu & TinyMLC Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is part of TinyMLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Host test entry - for local debugging on x86/x64/ARM64
// Uses standard printf and exit

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "model.h"

int main() {
    // Test inputs - initialized with constant value
    static int8_t input1[INPUT_SIZE_1];
    static int8_t output[OUTPUT_SIZE];

    printf("TinyMLC Host Test\n");
    printf("Input size: %d, Output size: %d\n", INPUT_SIZE_1, OUTPUT_SIZE);

    for (int i = 0; i < INPUT_SIZE_1; i++) {
        input1[i] = 1;
    }

    // Call inference
#if defined(INPUT_SIZE_2) && INPUT_SIZE_2 > 0
    static int8_t input2[INPUT_SIZE_2];
    printf("Input2 size: %d\n", INPUT_SIZE_2);
    for (int i = 0; i < INPUT_SIZE_2; i++) {
        input2[i] = 1;
    }
    tinymlc_inference(input1, input2, output);
#else
    tinymlc_inference(input1, output);
#endif

    // Output result
    printf("Output:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("ALL-OK\n");
    return 0;
}