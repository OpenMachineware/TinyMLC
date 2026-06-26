# For building NMSIS-NN
# cmake ../Source \
#   -DCMAKE_TOOLCHAIN_FILE=../riscv-none-elf-gcc.cmake \
#   -DRISCV_ARCH=rv32imac \
#   -DRISCV_ABI=ilp32 \
#   -DRISCV_MODEL=medany
# make
# Then copy Include and libNMSISNN.a

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)

set(CROSS_COMPILE "riscv-none-elf-")

set(CMAKE_C_COMPILER ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE}g++)
set(CMAKE_ASM_COMPILER ${CROSS_COMPILE}gcc)
set(CMAKE_AR ${CROSS_COMPILE}ar)
set(CMAKE_OBJCOPY ${CROSS_COMPILE}objcopy)
set(CMAKE_OBJDUMP ${CROSS_COMPILE}objdump)

set(RISCV_ARCH "rv32imac")
set(RISCV_ABI "ilp32")

add_compile_options(
    -march=${RISCV_ARCH}
    -mabi=${RISCV_ABI}
    -mcmodel=medany
    -O2
    -Wall
)
