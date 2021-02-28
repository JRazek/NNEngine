#pragma once

#include <vector>
#include "netStructure/Layer/Layer.h"
#include "dataStructures/Tensor.h"

struct Net{
    //{ layerNum:{type, neuronsSize, inputSize} }
    //{ layerNum:{type, tensorsCount, tensorDepth, matrixSizeX, matrixSizeY} }
    Net(std::vector<std::vector<int>> structure, int seed = 0);
    
    void run(const Tensor &tensorInput);
    std::vector<float> getResult(){
        Layer * last = layers[layers.size() - 1];
        return last->outputVector;
    }
    ~Net();
    std::vector<Layer *> layers;
};