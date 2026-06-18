.section .text
.globl _start
_start:
    li sp, 0x80004000
    call main
    j .