.section .text
.globl _start
_start:
    li sp, 0x80004000      # 初始化栈指针
    call main              # 调用 C 的 main 函数
    j .                    # 死循环
