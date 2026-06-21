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
