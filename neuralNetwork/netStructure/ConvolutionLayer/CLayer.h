#pragma once
#include "CLayer.h"
#include <netStructure/Layer/Layer.h>

struct CLayer : Layer{
    CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY);
    void run(const std::vector<float> &input);
    ~CLayer();
};