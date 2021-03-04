#pragma once
#include <utils/dataStructures/Tensor.h>
#include <netStructure/Layer/Layer.h>
struct PoolingLayer : Layer{
    const int kernelSizeX;
    const int kernelSizeY;
    PoolingLayer(int id, Net * net, int kernelSizeX, int kernelSizeY);
    ~PoolingLayer();
    void run(const Tensor &tensor);
    Tensor outputTensor;
};