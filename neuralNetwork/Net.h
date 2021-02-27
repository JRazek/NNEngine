#pragma once

#include <vector>
#include "netStructure/Layer/Layer.h"

struct Net{
    //{ layerNum:{type, neuronsSize/tensorsSize, inputSize} }
    Net(std::vector<std::vector<int>> structure, int seed = 0);
    const int settingsLayerTypeInd = 0;
    const int settingsLayerSizeInd = 1;
    const int settingsInputSizeInd = 2;

    void run(std::vector<float> input);
    ~Net();
    std::vector<Layer *> layers;
};