option(COMPILE_WITH_CUDA "Enable CUDA" OFF)

cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -Werror=vla")

if(COMPILE_WITH_CUDA)
    project(NeuralNetLibrary LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    add_library(ImageProcessingLib CUDA/CUDAUtils.cuh Network/layers/ConvolutionLayer/CUDAConvolutionLayer.cuh Network/layers/OutputLayer/CUDAOutputLayer.cuh CUDA/CUDAUtils.cu Network/layers/ConvolutionLayer/CUDAConvolutionLayer.cu Network/layers/OutputLayer/CUDAOutputLayer.cu Network/Network.cpp Network/Network.h Network/layers/interfaces/Layer.cpp Network/layers/interfaces/Layer.h Network/layers/FFLayer/FFLayer.cpp Network/layers/FFLayer/FFLayer.h Network/layers/ConvolutionLayer/ConvolutionLayer.cpp Network/layers/ConvolutionLayer/ConvolutionLayer.h Utils/dataStructures/Tensor.h structureTesting.cpp Utils/Utils.cpp Utils/Utils.h Utils/dataStructures/QuadTree.cpp Utils/dataStructures/QuadTree.h Utils/dataStructures/PointData.h Utils/dataStructures/PointData.cpp Utils/dataStructures/KDTree.cpp Utils/dataStructures/KDTree.h testGenerator.cpp convTesting.cpp Utils/dataStructures/TMatrix.h Utils/dataStructures/Vector2.h Network/layers/FlatteningLayer/FlatteningLayer.cpp Network/layers/FlatteningLayer/FlatteningLayer.h Utils/dataStructures/Vector3.h Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.cpp Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.h Network/layers/MaxPoolingLayer/MaxPoolingLayer.cpp Network/layers/MaxPoolingLayer/MaxPoolingLayer.h Optimizers/MBGD.cpp Optimizers/MBGD.h Network/layers/OutputLayer/OutputLayer.cpp Network/layers/OutputLayer/OutputLayer.h Network/layers/interfaces/Learnable.cpp Network/layers/interfaces/Learnable.h Utils/BitmapEncodings.h Utils/dataStructures/PrefixSum2D.h Utils/Files/CSVReader.cpp Utils/Files/CSVReader.h Utils/Files/ImageRepresentation.cpp Utils/Files/ImageRepresentation.h Optimizers/MomentumGD.cpp Optimizers/MomentumGD.h Optimizers/interfaces/Optimizer.cpp Optimizers/interfaces/Optimizer.h Utils/interfaces/JSONEncodable.h Network/layers/ActivationLayers/ReLU/ReLU.cpp Network/layers/ActivationLayers/ReLU/ReLU.h Network/layers/ActivationLayers/Sigmoid/Sigmoid.cpp Network/layers/ActivationLayers/Sigmoid/Sigmoid.h Utils/Files/NetFileManager.cpp Utils/Files/NetFileManager.h Network/layers/InputLayer/InputLayer.cpp Network/layers/InputLayer/InputLayer.h  Network/layers/RecurrentLayer/RecurrentLayer.cpp Network/layers/RecurrentLayer/RecurrentLayer.h Utils/dataStructures/VectorN.h Utils/dataStructures/Vector4.h)
    target_compile_definitions(ImageProcessingLib PUBLIC NNL_WITH_CUDA)
else()
    project(NeuralNetLibrary LANGUAGES CXX)
    add_library(ImageProcessingLib CUDA/CUDAUtils.cuh Network/layers/ConvolutionLayer/CUDAConvolutionLayer.cuh Network/layers/OutputLayer/CUDAOutputLayer.cuh Network/Network.cpp Network/Network.h Network/layers/interfaces/Layer.cpp Network/layers/interfaces/Layer.h Network/layers/FFLayer/FFLayer.cpp Network/layers/FFLayer/FFLayer.h Network/layers/ConvolutionLayer/ConvolutionLayer.cpp Network/layers/ConvolutionLayer/ConvolutionLayer.h Utils/dataStructures/Tensor.h structureTesting.cpp Utils/Utils.cpp Utils/Utils.h Utils/dataStructures/QuadTree.cpp Utils/dataStructures/QuadTree.h Utils/dataStructures/PointData.h Utils/dataStructures/PointData.cpp Utils/dataStructures/KDTree.cpp Utils/dataStructures/KDTree.h testGenerator.cpp convTesting.cpp Utils/dataStructures/TMatrix.h Utils/dataStructures/Vector2.h Network/layers/FlatteningLayer/FlatteningLayer.cpp Network/layers/FlatteningLayer/FlatteningLayer.h Utils/dataStructures/Vector3.h Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.cpp Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.h Network/layers/MaxPoolingLayer/MaxPoolingLayer.cpp Network/layers/MaxPoolingLayer/MaxPoolingLayer.h Optimizers/MBGD.cpp Optimizers/MBGD.h Network/layers/OutputLayer/OutputLayer.cpp Network/layers/OutputLayer/OutputLayer.h Network/layers/interfaces/Learnable.cpp Network/layers/interfaces/Learnable.h Utils/BitmapEncodings.h Utils/dataStructures/PrefixSum2D.h Utils/Files/CSVReader.cpp Utils/Files/CSVReader.h Utils/Files/ImageRepresentation.cpp Utils/Files/ImageRepresentation.h Optimizers/MomentumGD.cpp Optimizers/MomentumGD.h Optimizers/interfaces/Optimizer.cpp Optimizers/interfaces/Optimizer.h Utils/interfaces/JSONEncodable.h Network/layers/ActivationLayers/ReLU/ReLU.cpp Network/layers/ActivationLayers/ReLU/ReLU.h Network/layers/ActivationLayers/Sigmoid/Sigmoid.cpp Network/layers/ActivationLayers/Sigmoid/Sigmoid.h Utils/Files/NetFileManager.cpp Utils/Files/NetFileManager.h Network/layers/InputLayer/InputLayer.cpp Network/layers/InputLayer/InputLayer.h  Network/layers/RecurrentLayer/RecurrentLayer.cpp Network/layers/RecurrentLayer/RecurrentLayer.h Utils/dataStructures/VectorN.h Utils/dataStructures/Vector4.h)
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(ImageProcessingLib PRIVATE nlohmann_json::nlohmann_json)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(Main convTesting.cpp)


target_link_libraries(Main ImageProcessingLib)
target_link_libraries(Main ${OpenCV_LIBS})