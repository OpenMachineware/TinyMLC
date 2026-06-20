ENTRY(_start)
MEMORY {
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 64K
}

SECTIONS {
    .text : {
        *(.vector_table) /* 向量表必须放在最前面 */
        *(.text*)
        *(.rodata*)
    } > FLASH

    .data : {
        _sdata = .;
        *(.data*)
        _edata = .;
    } > RAM AT > FLASH
    _sidata = LOADADDR(.data); /* 记录 .data 在 Flash 中的加载地址 */

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > RAM

    /* 在 RAM 末尾分配栈空间，严格 8 字节对齐 */
    . = ALIGN(8);
    . = . + 0x1000; /* 分配 4KB 栈空间，防止函数调用溢出 */
    _estack = .;
}
