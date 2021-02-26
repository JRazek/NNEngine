#include "CLayer.h"
#include <Net.h>

CLayer::CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY):Layer(id, net){}
CLayer::~CLayer(){}