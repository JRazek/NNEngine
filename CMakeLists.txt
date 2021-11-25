option(COMPILE_WITH_CUDA "Enable CUDA" OFF)
option(COMPILE_TEST_PROGRAM "Test Program" OFF)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -Werror=vla")

add_subdirectory(Utils/dataStructures)

set(LIBRARY_FILES Optimizers/GeneticOptimizer.cpp Network/layers/ActivationLayers/Softmax/Softmax.cpp Network/layers/RecurrentLayer/RecurrentOutputLayer/RecurrentOutputLayer.cpp Network/Network.cpp Network/layers/FFLayer/FFLayer.cpp Network/layers/ConvolutionLayer/ConvolutionLayer.cpp Utils/Utils.cpp Network/layers/FlatteningLayer/FlatteningLayer.cpp Network/layers/FlatteningLayer/FlatteningLayer.h Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.cpp Network/layers/MaxPoolingLayer/MaxPoolingLayer.cpp Optimizers/MBGD.cpp Optimizers/MBGD.h Network/layers/OutputLayer/OutputLayer.cpp Network/layers/interfaces/Learnable.cpp Utils/Files/CSVReader.cpp Utils/Files/ImageRepresentation.cpp Optimizers/MomentumGD.cpp Optimizers/interfaces/GradientOptimizer.cpp Network/layers/ActivationLayers/ReLU/ReLU.cpp Network/layers/ActivationLayers/Sigmoid/Sigmoid.cpp Utils/Files/NetFileManager.cpp Network/layers/InputLayer/InputLayer.cpp Network/layers/RecurrentLayer/RecurrentLayer.cpp)
set(CUDA_FILES Network/layers/OutputLayer/CUDAOutputLayer.cuh CUDA/CUDAUtils.cu Network/layers/ConvolutionLayer/CUDAConvolutionLayer.cu Network/layers/OutputLayer/CUDAOutputLayer.cu)

project(NeuralFlows LANGUAGES CXX)
add_library(NeuralFlows SHARED ${LIBRARY_FILES})

if(COMPILE_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    target_sources(NeuralFlows PUBLIC ${CUDA_FILES})
    target_compile_definitions(NeuralFlows PUBLIC NNL_WITH_CUDA)
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(NeuralFlows PRIVATE nlohmann_json::nlohmann_json)

if(COMPILE_TEST_PROGRAM)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(Main convTesting.cpp)
    target_link_libraries(Main NeuralFlows)
    target_link_libraries(Main ${OpenCV_LIBS})
endif()


install(TARGETS NeuralFlows)
install(
        DIRECTORY "${CMAKE_SOURCE_DIR}"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
        PATTERN "build" EXCLUDE
        PATTERN "cmake-build-*" EXCLUDE
        PATTERN ".git" EXCLUDE
)