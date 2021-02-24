cmake_minimum_required(VERSION 3.12)
project(NeuralNetLibrary CXX)

add_subdirectory(neuralNetwork)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC neuralNetwork)
target_link_directories(${PROJECT_NAME} PUBLIC neuralNetwork)

target_link_libraries(${PROJECT_NAME} PUBLIC Layer FFLayer ConvolutionLayer Net)