.syntax unified
.cpu cortex-m4
.thumb

.global _start
.section .text
_start:
    /* 设置栈指针（QEMU virt 板 RAM 起始地址） */
    ldr sp, =0x20004000

    /* 清零 .bss 段 */
    ldr r0, =_sbss
    ldr r1, =_ebss
    subs r1, r1, r0
    beq 2f
1:
    movs r2, #0
    strb r2, [r0], #1
    subs r1, r1, #1
    bne 1b
2:

    /* 调用 main */
    bl main

    /* 死循环 */
    b .
