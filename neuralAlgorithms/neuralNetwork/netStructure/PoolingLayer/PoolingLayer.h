#pragma once
#include <utils/dataStructures/Tensor.h>
struct PoolingLayer{
    const int kernelSizeX;
    const int kernelSizeY;
    PoolingLayer(int kernelSizeX, int kernelSizeY);
    ~PoolingLayer();
    void run(const Tensor &tensor);
    Tensor * outputTensor;
};