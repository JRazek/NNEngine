#pragma once
#include "CLayer.h"
#include <netStructure/Layer/Layer.h>

struct CLayer : Layer{
    CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY);
    ~CLayer();
};