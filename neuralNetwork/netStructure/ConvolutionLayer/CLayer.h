#pragma once
#include <netStructure/Layer/Layer.h>
#include <dataStructures/Tensor.h>

struct CLayer : Layer{
    CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY);
    void run(const Tensor &input);
    void initWeights();
    std::vector<std::pair< Tensor, float >> tensors;//tensor, bias

    ~CLayer();
};