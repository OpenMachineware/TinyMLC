// Host debug print - uses standard printf

#include <stdio.h>
#include <stdint.h>
#include <string.h>

void tinymlc_putchar(char c) {
    putchar(c);
}

void tinymlc_print_int(int n) {
    printf("%d", n);
}

void debug_char(char c) {
    putchar(c);
}

void debug_str(const char* str) {
    printf("%s", str);
}

void debug_int(int n) {
    printf("%d", n);
}

void debug_hex(unsigned int n) {
    printf("0x%08X", n);
}

void debug_endl(void) {
    putchar('\n');
}