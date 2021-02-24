#pragma once

#include <vector>
#include "netStructure/FFLayer/FFLayer.h"

struct Net{
    //int for type, int for neuronsCount if applicable
    Net(std::vector<std::pair<int, int>> structure, int seed = 0);
    ~Net();
    std::vector<Layer *> layers;
};