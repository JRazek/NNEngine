cmake_minimum_required(VERSION 3.12)
project(Main CXX)

add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(neuralAlgorithms)

target_link_libraries(${PROJECT_NAME} PUBLIC Net)
target_link_libraries(${PROJECT_NAME} PUBLIC GeneticLearningUnit)
#target_link_libraries(${PROJECT_NAME} PUBLIC NeuralAlgorithms)