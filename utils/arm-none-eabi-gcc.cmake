# 编译CMSIS-NN用的
# cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-none-eabi-gcc.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/cmsis-nn
# make
# 然后拷贝Include和libcmsis-nn.a就行

# 设置目标系统为裸机环境，处理器架构为 ARM
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

# 指定交叉编译器
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP arm-none-eabi-objdump)
set(CMAKE_SIZE arm-none-eabi-size)

# 禁止 CMake 尝试编译并运行测试程序（因为交叉编译的二进制无法在宿主机上运行）
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# 设置查找路径规则，防止误引入宿主机的库和头文件
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
