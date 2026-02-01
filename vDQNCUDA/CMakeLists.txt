cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(DQNcuda_Project LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找 LibTorch (PyTorch)
# 使用时需要在命令行指定 -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 2. 寻找 OpenCV
find_package(OpenCV REQUIRED)

# 3. 寻找 CUDA (虽然 LibTorch 自带 CUDA，但你代码里直接用了 cuda_runtime.h)
find_package(CUDA REQUIRED)

# 4. 定义可执行文件
# 假设你的文件名是 DQNcuda.cpp
add_executable(DQNcuda DQNcuda.cpp)

# 5. 链接库
# 注意：torch_cuda 提供了 CUDA 支持
target_link_libraries(DQNcuda 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# 6. CUDA 相关设置
# 如果需要指定显卡架构（例如 RTX 30系列是 86，RTX 40系列是 89）
set_target_properties(DQNcuda PROPERTIES CUDA_ARCHITECTURES 86)

# 解决 LibTorch 可能导致的 ABI 问题
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET DQNcuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:DQNcuda>)
endif()
