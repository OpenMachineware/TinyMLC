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
