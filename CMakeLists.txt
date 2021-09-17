
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
set(CMAKE_CXX_COMPILER /usr/bin/gcc)
cmake_minimum_required(VERSION 3.0.2)
project(NeuralNetLibrary LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(nlohmann_json 3.2.0 REQUIRED)

#add_executable(Test structureTesting.cpp)
#add_executable(TestGenerator testGenerator.cpp)
add_executable(Main convTesting.cpp)


add_library(ImageProcessingLib Network/Network.cpp Network/Network.h Network/layers/interfaces/Layer.cpp Network/layers/interfaces/Layer.h Network/layers/FFLayer/FFLayer.cpp Network/layers/FFLayer/FFLayer.h Network/layers/ConvolutionLayer/ConvolutionLayer.cpp Network/layers/ConvolutionLayer/ConvolutionLayer.h Utils/dataStructures/Bitmap.h structureTesting.cpp Utils/Utils.cpp Utils/Utils.h Utils/dataStructures/QuadTree.cpp Utils/dataStructures/QuadTree.h Utils/dataStructures/PointData.h Utils/dataStructures/PointData.cpp Utils/dataStructures/KDTree.cpp Utils/dataStructures/KDTree.h testGenerator.cpp convTesting.cpp Utils/dataStructures/TMatrix.h Utils/dataStructures/Vector2.h Network/layers/FlatteningLayer/FlatteningLayer.cpp Network/layers/FlatteningLayer/FlatteningLayer.h Utils/dataStructures/Vector3.h Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.cpp Network/layers/BatchNormalizationLayer/BatchNormalizationLayer.h Network/layers/MaxPoolingLayer/MaxPoolingLayer.cpp Network/layers/MaxPoolingLayer/MaxPoolingLayer.h Optimizers/MBGD.cpp Optimizers/MBGD.h Network/layers/OutputLayer/OutputLayer.cpp Network/layers/OutputLayer/OutputLayer.h Network/layers/interfaces/Learnable.cpp Network/layers/interfaces/Learnable.h Utils/BitmapEncodings.h Utils/dataStructures/PrefixSum2D.h Utils/Files/CSVReader.cpp Utils/Files/CSVReader.h Utils/Files/ImageRepresentation.cpp Utils/Files/ImageRepresentation.h Optimizers/MomentumGD.cpp Optimizers/MomentumGD.h Optimizers/interfaces/Optimizer.cpp Optimizers/interfaces/Optimizer.h Utils/interfaces/JSONEncodable.h Network/layers/ActivationLayers/ReLU/ReLU.cpp Network/layers/ActivationLayers/ReLU/ReLU.h Network/layers/ActivationLayers/Sigmoid/Sigmoid.cpp Network/layers/ActivationLayers/Sigmoid/Sigmoid.h Utils/Files/NetFileManager.cpp Utils/Files/NetFileManager.h Network/layers/InputLayer/InputLayer.cpp Network/layers/InputLayer/InputLayer.h CUDA/CUDAUtils.cu CUDA/CUDAUtils.cuh Network/layers/ConvolutionLayer/CUDAConvolutionLayer.cu Network/layers/ConvolutionLayer/CUDAConvolutionLayer.cuh)
target_link_libraries(ImageProcessingLib PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(Main ImageProcessingLib)
target_link_libraries(Main ${OpenCV_LIBS})
target_link_libraries(Main ${OpenCV_LIBS})