cmake_minimum_required(VERSION 3.0.2)
project(NeuralNetLibrary CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Test structureTesting.cpp)
add_executable(TestGenerator testGenerator.cpp)
add_executable(Main convTesting.cpp)

add_library(ConvolutionNetLib Network/Network.cpp Network/Network.h Network/layers/interfaces/Layer.cpp Network/layers/interfaces/Layer.h Network/layers/FFLayer.cpp Network/layers/FFLayer.h Network/layers/ConvolutionLayer.cpp Network/layers/ConvolutionLayer.h Utils/Bitmap.h structureTesting.cpp Utils/Utils.cpp Utils/Utils.h Utils/dataStructures/QuadTree.cpp Utils/dataStructures/QuadTree.h Utils/dataStructures/PointData.h Utils/dataStructures/PointData.cpp Utils/dataStructures/KDTree.cpp Utils/dataStructures/KDTree.h testGenerator.cpp convTesting.cpp Utils/dataStructures/TMatrix.h Utils/dataStructures/Vector2.h Utils/Differentiables/DifferentiableFunction.h Utils/Differentiables/ReLU.cpp Utils/Differentiables/ReLU.h Utils/Differentiables/Sigmoid.cpp Utils/Differentiables/Sigmoid.h Network/layers/FlatteningLayer.cpp Network/layers/FlatteningLayer.h Utils/dataStructures/Vector3.h Network/layers/BatchNormalizationLayer.cpp Network/layers/BatchNormalizationLayer.h Network/layers/MaxPoolingLayer.cpp Network/layers/MaxPoolingLayer.h LearningModels/Backpropagation.cpp LearningModels/Backpropagation.h Network/layers/OutputLayer.cpp Network/layers/OutputLayer.h Network/layers/interfaces/Learnable.cpp Network/layers/interfaces/Learnable.h)

target_link_libraries(Main ConvolutionNetLib)
target_link_libraries(Main ${OpenCV_LIBS})