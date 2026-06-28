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

#include "debug_print.h"

#include <stdint.h>

#define UART_BASE 0x40004000
#define UART_DR (*(volatile uint32_t *)(UART_BASE + 0x00))
#define UART_STATE (*(volatile uint32_t *)(UART_BASE + 0x04))
#define UART_CTRL (*(volatile uint32_t *)(UART_BASE + 0x08))
#define UART_BAUDDIV (*(volatile uint32_t *)(UART_BASE + 0x10))

#define UART_STATE_TXBF (1 << 0)

void uart_init(void) {
    UART_BAUDDIV = 13;
    UART_CTRL = 0x3;
}

void tinymlc_putchar(char c) {
    while (UART_STATE & UART_STATE_TXBF);
    UART_DR = c;
}

void tinymlc_print_int(int n) {
    if (n < 0) {
        tinymlc_putchar('-');
        n = -n;
    }
    if (n >= 10) {
        tinymlc_print_int(n / 10);
    }
    tinymlc_putchar('0' + (n % 10));
}

void debug_char(char c) {
    while (UART_STATE & UART_STATE_TXBF);
    UART_DR = c;
}

void debug_str(const char* str) {
    while (*str) {
        debug_char(*str++);
    }
}

void debug_int(int n) {
    if (n < 0) {
        debug_char('-');
        n = -n;
    }
    if (n >= 10) {
        debug_int(n / 10);
    }
    debug_char('0' + (n % 10));
}

void debug_hex(unsigned int n) {
    const char hex[] = "0123456789ABCDEF";
    // Print 8-digit hex with leading zeros
    for (int i = 7; i >= 0; i--) {
        debug_char(hex[(n >> (i * 4)) & 0xF]);
    }
}

void debug_endl(void) {
    debug_char('\n');
}

void* memset(void* s, int c, unsigned int n) {
    unsigned char* p = (unsigned char*)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

void* memcpy(void* dest, const void* src, unsigned int n) {
    unsigned char* d = (unsigned char*)dest;
    const unsigned char* s = (const unsigned char*)src;
    while (n--) *d++ = *s++;
    return dest;
}
