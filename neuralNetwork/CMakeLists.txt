cmake_minimum_required(VERSION 3.12)
project(NeuralNetLibrary CXX)

add_library(
    Net
    Net.h
    Net.cpp
    netStructure/ConvolutionLayer/CLayer.h
    netStructure/ConvolutionLayer/CLayer.cpp
    netStructure/FFLayer/FFLayer.h
    netStructure/FFLayer/FFLayer.cpp
    netStructure/Layer/Layer.h
    activations/ActivationFunction.h
    activations/SigmoidFunction.h
    dataStructures/Matrix.h
    dataStructures/Matrix.cpp
    dataStructures/Tensor.h
    dataStructures/Tensor.cpp
)
include_directories(./)