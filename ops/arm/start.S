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

.syntax unified
.cpu cortex-m4
.thumb

/* Declare symbols defined in linker script */
.extern _estack, _sbbs, _ebss

.section .vector_table, "a"
.global _start
_start:
    .word _estack           /* 0x00: Initial Main Stack Pointer (MSP) */
    .word Reset_Handler     /* 0x04: Reset Handler */

.section .text.Reset_Handler
.global Reset_Handler
.type Reset_Handler, %function
Reset_Handler:
    /* 1. Explicitly set stack pointer (double safety) */
    LDR r0, =_estack
    MOV sp, r0

    /* 2. Clear .bss section (uninitialized global variables) */
    LDR r0, =_sbss
    LDR r1, =_ebss
    MOV r2, #0
bss_loop:
    CMP r0, r1
    BGE bss_done
    STR r2, [r0], #4
    B bss_loop
bss_done:

    /* 3. Jump to C main function */
    BL main

    /* 4. Prevent runaway after main returns */
hang:
    B hang

/* Force literal pool generation to prevent offset overflow errors */
.ltorg
