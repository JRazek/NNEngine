#include "CLayer.h"
#include <Net.h>
#include <iostream>
//todo
CLayer::CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY):Layer(id, net, 0){}//blank 0
void CLayer::run(const std::vector<float> &input){
    std::cout<<"Im a fucking CLayer!\n";
}
CLayer::~CLayer(){}

CLayer::Tensor::Tensor(int x, int y, int z):matrices(z){
    for(int i = 0; i < z; i ++){
        matrices.push_back(Matrix(x, y));
    }
}
CLayer::Tensor::Matrix::Matrix(int sizeX, int sizeY){
    for(int y = 0; y < sizeY; y++){
        weights.push_back(std::vector<float>());
        for(int x = 0; x < sizeX; x++){
            weights[y].push_back(0);
        }
    }
}