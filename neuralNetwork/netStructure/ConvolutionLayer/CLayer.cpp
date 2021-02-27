#include "CLayer.h"
#include <Net.h>
#include <iostream>
//todo
CLayer::CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY):Layer(id, net, 0){
    for(int i = 0; i < tensorCount; i ++){
        Kernel tensor = Kernel(tensorDepth, matrixSizeX, matrixSizeY);
        for(int z = 0; z < tensorDepth; z ++){
            tensors[i].matrices.push_back(CLayer::Kernel::Matrix(matrixSizeX, matrixSizeY));
        }
        this->tensors.push_back(tensor);
    }
}//blank 0
void CLayer::initWeights(){
    for(int i = 0; i < this->tensors.size(); i ++){
        float randBias = (rand() % 1000)/100.f;
        tensors[i].bias = randBias;
        for(int z = 0; z < tensors[i].matrices.size(); z ++){
            for(int y = 0; y < tensors[i].matrices[z].weights.size(); y ++){
                for(int x = 0; x < tensors[i].matrices[z].weights[y].size(); x ++){
                    float randWeight = (rand() % 100)/100.f;
                    tensors[i].matrices[z].weights[y][x] = randWeight;
                }
            }
        }
    }
}
void CLayer::run(const std::vector<float> &input){
    std::cout<<"Im a fucking CLayer!\n";
}
CLayer::~CLayer(){}

CLayer::Kernel::Kernel(int x, int y, int z){
    for(int i = 0; i < z; i ++){
        matrices.push_back(Matrix(x, y));
    } 
}
CLayer::Kernel::Matrix::Matrix(int sizeX, int sizeY){
    for(int y = 0; y < sizeY; y++){
        weights.push_back(std::vector<float>());
        for(int x = 0; x < sizeX; x++){
            weights[y].push_back(0);
        }
    }
}