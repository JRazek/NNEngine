#pragma once
#include <netStructure/Layer/Layer.h>
#include <utils/dataStructures/Tensor.h>

struct CLayer : Layer{
    CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY);
    void run(const Tensor &input);
    void initWeights();
    const int kernelSizeX, kernelSizeY, kernelSizeZ;
    std::vector<std::pair< Tensor, float >> tensors;//tensor, bias
    const int stride;
    const int padding;

    Tensor * outputTensor;

    ~CLayer();
};