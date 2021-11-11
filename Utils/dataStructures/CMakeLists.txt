cmake_minimum_required(VERSION 3.16)
project(DataStructures)

set(CMAKE_CXX_STANDARD 20)

add_library(DataStructures VectorN.h Vector2.h Vector3.h Vector4.h TMatrix.h Tensor.h QuadTree.cpp PointData.cpp KDTree.cpp PrefixSum2D.h)
