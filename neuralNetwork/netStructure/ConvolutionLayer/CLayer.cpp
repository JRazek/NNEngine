#include "CLayer.h"
#include <Net.h>
#include <iostream>
//todo
CLayer::CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY):Layer(id, net, 0){}//blank 0
void CLayer::run(const std::vector<float> &input){
    std::cout<<"Im a fucking CLayer!\n";
}
CLayer::~CLayer(){}