cmake_minimum_required(VERSION 3.10)
project(cpp_robot_arm)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找pybind11和Eigen3
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# 添加源文件
pybind11_add_module(cpp_robot_arm robot_arm_binding.cpp)

# 添加头文件路径
target_include_directories(cpp_robot_arm PRIVATE ${EIGEN3_INCLUDE_DIR})

# 链接Eigen库
target_link_libraries(cpp_robot_arm PRIVATE Eigen3::Eigen)
