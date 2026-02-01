cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(CartPoleDQN)

# 1. 设置 C++ 标准 (LibTorch 至少需要 C++14，推荐 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 查找依赖库
# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 LibTorch (PyTorch C++)
# 如果 CMake 找不到路径，请在命令行使用 -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

# 3. 添加头文件路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# 4. 定义可执行文件
add_executable(dqn_cartpole source.cpp)

# 5. 链接库
target_link_libraries(dqn_cartpole 
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES}
)

# 6. 针对 Windows (MSVC) 的特殊配置
if (MSVC)
    # 解决 LibTorch 与 MSVC 的部分符号冲突及编译选项
    target_compile_options(dqn_cartpole PUBLIC /utf-8)
    if (NOT MSVC_VERSION LESS 1900)
        # 启用多核编译加速
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        target_compile_options(dqn_cartpole PUBLIC /MP)
    endif()
endif()

# 如果使用 GPU 版本，请确保设置以下属性
set_property(TARGET dqn_cartpole PROPERTY CXX_STANDARD 17)