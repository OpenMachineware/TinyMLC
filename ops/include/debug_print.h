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

#ifndef DEBUG_PRINT_H
#define DEBUG_PRINT_H

#include <stdint.h>

// Debug macro switch
// Enable debug by compiling with -DTINYMLC_DEBUG
#ifdef TINYMLC_DEBUG
    // Debug mode: enable printing
    void tinymlc_putchar(char c);
    void tinymlc_print_int(int n);
    void debug_char(char c);
    void debug_str(const char* str);
    void debug_int(int n);
    void debug_hex(unsigned int n);
    void debug_endl(void);
    void uart_init(void);

    #define TMLC_PUTCHAR(c) tinymlc_putchar(c)
    #define TMLC_PRINT_INT(n) tinymlc_print_int(n)
    #define DEBUG_CHAR(c) debug_char(c)
    #define DEBUG_STR(s) debug_str(s)
    #define DEBUG_INT(n) debug_int(n)
    #define DEBUG_HEX(n) debug_hex(n)
    #define DEBUG_ENDL() debug_endl()

    // Quick position marker: print filename and line number
    #define DEBUG_POS() do { \
        debug_str(__FILE__); \
        debug_char(':'); \
        debug_int(__LINE__); \
        debug_endl(); \
    } while(0)

#else
    // Release mode: macros expand to empty
    #define TMLC_PUTCHAR(c) ((void)0)
    #define TMLC_PRINT_INT(n) ((void)0)
    #define DEBUG_CHAR(c) ((void)0)
    #define DEBUG_STR(s) ((void)0)
    #define DEBUG_INT(n) ((void)0)
    #define DEBUG_HEX(n) ((void)0)
    #define DEBUG_ENDL() ((void)0)
    #define DEBUG_POS() ((void)0)
#endif

#endif // DEBUG_PRINT_H
