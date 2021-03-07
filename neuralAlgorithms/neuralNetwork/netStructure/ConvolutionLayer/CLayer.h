#pragma once
#include <netStructure/Layer/Layer.h>
#include <utils/dataStructures/Tensor.h>
#include <activations/ActivationFunction.h>

struct CLayer : Layer{
    CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY, ActivationFunction * activationFunction);
    CLayer(int id, Net * net, const CLayer &p1, const CLayer &p2);
    void run(const Tensor &input);
    void initWeights();
    const int kernelSizeX, kernelSizeY, kernelSizeZ;
    std::vector<std::pair< Tensor, float >> tensors;//tensor, bias
    const int stride;
    const int padding;
    ActivationFunction * activationFunction;

    Tensor outputTensor;

    ~CLayer();
};