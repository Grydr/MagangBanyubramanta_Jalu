cmake_minimum_required(VERSION 3.5)
project(bt_try)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(behaviortree_cpp_v3 REQUIRED)

add_executable(bt_try BT.cpp)
target_link_libraries(bt_try BT::behaviortree_cpp_v3)