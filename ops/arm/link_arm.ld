ENTRY(_start)
MEMORY {
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 64K
}

SECTIONS {
    .text : {
        *(.vector_table) /* Vector table must be placed first */
        *(.text*)
        *(.rodata*)
    } > FLASH

    .data : {
        _sdata = .;
        *(.data*)
        _edata = .;
    } > RAM AT > FLASH
    _sidata = LOADADDR(.data); /* Record .data load address in Flash */

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > RAM

    /* Allocate stack space at end of RAM, strictly 8-byte aligned */
    . = ALIGN(8);
    . = . + 0x1000; /* Allocate 4KB stack space to prevent call overflow */
    _estack = .;
}
