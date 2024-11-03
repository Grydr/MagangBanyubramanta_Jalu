cmake_minimum_required(VERSION 3.5)
project(my_bt)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(behaviortree_cpp_v3 REQUIRED)

add_executable(key_logger key.cpp)
target_link_libraries(key_logger BT::behaviortree_cpp_v3)