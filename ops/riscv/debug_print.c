#include "debug_print.h"

// UART address (matches QEMU virt board)
#define UART_TX_ADDR ((volatile char*)0x10000000)

void tinymlc_putchar(char c) {
    *UART_TX_ADDR = c;
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
    *UART_TX_ADDR = c;
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
