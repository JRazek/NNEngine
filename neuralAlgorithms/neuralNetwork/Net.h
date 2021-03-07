#pragma once

#include <vector>
#include "netStructure/Layer/Layer.h"
#include "utils/dataStructures/Tensor.h"

struct Net{
    //{ layerNum:{type, neuronsSize, inputSize} }
    //{ layerNum:{type, tensorsCount, matrixSizeX, matrixSizeY, tensorDepth} }
    Net(std::vector<std::vector<int>> structure, int seed = 0);

    //Net(std::vector<Layer *> &layers);
    Net();
    void run(const std::vector<float> &inputVector);
    void run(Tensor tensorInput);
    std::vector<float> getResult(){
        Layer * last = layers[layers.size() - 1];
        return last->outputVector;
    }
    ~Net();
    std::vector<Layer *> layers;
};