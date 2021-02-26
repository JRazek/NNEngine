#include "CLayer.h"
#include <Net.h>
#include <iostream>

CLayer::CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY):Layer(id, net){}
void CLayer::run(){
    std::cout<<"Im a fucking CLayer!\n";
}
CLayer::~CLayer(){}