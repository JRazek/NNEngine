#pragma once

#include <vector>
#include "netStructure/FFLayer/FFLayer.h"

struct Net{
    Net(std::vector<std::pair<bool, int>> structure);
    ~Net();
    std::vector<Layer *> layers;
};