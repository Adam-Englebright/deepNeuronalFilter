cmake_minimum_required(VERSION 3.8.0)
project(directional_audio)
set(CMAKE_CXX_STANDARD 17)
include_directories(..)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_library(alsa_cpp_wrapper alsa_cpp_wrapper.cpp)
target_link_libraries(alsa_cpp_wrapper -lasound)

add_executable(directional_audio directional_audio.cpp)
target_link_libraries(directional_audio alsa_cpp_wrapper dnf fir)
