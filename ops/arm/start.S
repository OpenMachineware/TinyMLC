.syntax unified
.cpu cortex-m4
.thumb

/* 声明链接脚本中定义的符号 */
.extern _estack, _sbss, _ebss

.section .vector_table, "a"
.global _start
_start:
    .word _estack           /* 0x00: 初始主栈指针 (MSP) */
    .word Reset_Handler     /* 0x04: 复位处理程序 */

.section .text.Reset_Handler
.global Reset_Handler
.type Reset_Handler, %function
Reset_Handler:
    /* 1. 显式设置栈指针（双重保险） */
    LDR r0, =_estack
    MOV sp, r0

    /* 2. 清零 .bss 段（未初始化的全局变量） */
    LDR r0, =_sbss
    LDR r1, =_ebss
    MOV r2, #0
bss_loop:
    CMP r0, r1
    BGE bss_done
    STR r2, [r0], #4
    B bss_loop
bss_done:

    /* 3. 跳转到 C 语言主函数 */
    BL main

    /* 4. 防止 main 返回后跑飞 */
hang:
    B hang

/* 强制生成字面量池，防止偏移量过大报错 */
.ltorg
