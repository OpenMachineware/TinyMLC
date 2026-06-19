OUTPUT_ARCH(arm)
ENTRY(_start)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1M
    RAM (rw)    : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
    .text : {
        *(.text*)
        *(.rodata*)
    } > FLASH

    .data : {
        *(.data*)
    } > RAM AT > FLASH

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM
}
