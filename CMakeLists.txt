cmake_minimum_required(VERSION 3.0.2)
project(NeuralNetLibrary CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Test main.cpp)

add_library(ConvolutionNetLib Network/Network.cpp Network/Network.h Network/layers/Layer.cpp Network/layers/Layer.h Network/layers/FFLayer.cpp Network/layers/FFLayer.h Network/layers/ConvolutionLayer.cpp Network/layers/ConvolutionLayer.h Network/Bitmap.cpp Network/Bitmap.h main.cpp)

target_link_libraries(Test ConvolutionNetLib)
target_link_libraries(Test ${OpenCV_LIBS})