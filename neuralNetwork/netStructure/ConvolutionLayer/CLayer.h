#pragma once
#include "../Layer/Layer.h"

struct CLayer : Layer{
    CLayer(int id, int kernelSizeX, int kernelSizeY);
    ~CLayer();
};