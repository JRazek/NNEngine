cmake_minimum_required(VERSION 3.0.2)
project(NeuralNetLibrary CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(ConvolutionNetLib Network/Network.cpp Network/Network.h Network/layers/Layer.cpp Network/layers/Layer.h Network/layers/FFLayer.cpp Network/layers/FFLayer.h Network/layers/ConvolutionLayer.cpp Network/layers/ConvolutionLayer.h)