OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM_CODE (rx)  : ORIGIN = 0x80000000, LENGTH = 16M
    RAM_DATA (rw)  : ORIGIN = 0x81000000, LENGTH = 16M
}

SECTIONS
{
    .text : { *(.text*) } > RAM_CODE
    .rodata : { *(.rodata*) } > RAM_DATA
    .data : { *(.data*) } > RAM_DATA
    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM_DATA
}
