#pragma once
#include <utils/dataStructures/Tensor.h>
#include <utils/dataStructures/Matrix.h>
struct Functions{
    static Matrix convolve(const Tensor &inputTensor, const Tensor &kernel, const int padding, const int stride);
};