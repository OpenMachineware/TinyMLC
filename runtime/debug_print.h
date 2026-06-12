#ifndef DEBUG_PRINT_H
#define DEBUG_PRINT_H

#include <stdint.h>

// 调试宏开关
// 编译时通过 -DTINYMLC_DEBUG 启用调试
#ifdef TINYMLC_DEBUG
    // 调试模式：启用打印
    void debug_char(char c);
    void debug_str(const char* str);
    void debug_int(int n);
    void debug_hex(unsigned int n);
    void debug_endl(void);  // 打印换行

    #define DEBUG_CHAR(c) debug_char(c)
    #define DEBUG_STR(s) debug_str(s)
    #define DEBUG_INT(n) debug_int(n)
    #define DEBUG_HEX(n) debug_hex(n)
    #define DEBUG_ENDL() debug_endl()

    // 快速标记位置：打印文件名和行号
    #define DEBUG_POS() do { \
        debug_str(__FILE__); \
        debug_char(':'); \
        debug_int(__LINE__); \
        debug_endl(); \
    } while(0)

#else
    // Release 模式：宏展开为空
    #define DEBUG_CHAR(c) ((void)0)
    #define DEBUG_STR(s) ((void)0)
    #define DEBUG_INT(n) ((void)0)
    #define DEBUG_HEX(n) ((void)0)
    #define DEBUG_ENDL() ((void)0)
    #define DEBUG_POS() ((void)0)
#endif

#endif // DEBUG_PRINT_H
