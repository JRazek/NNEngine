#include "CLayer.h"
#include <utils/Functions.h>
#include <Net.h>
#include <iostream>
//todo
CLayer::CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY):
    kernelSizeX(matrixSizeX), kernelSizeY(matrixSizeY), kernelSizeZ(tensorDepth), stride(1), padding(0),
    Layer(id, net, 0){
        for(int i = 0; i < tensorCount; i ++){
            Tensor tensor = Tensor(tensorDepth, matrixSizeX, matrixSizeY);
            this->tensors.push_back({tensor, 0});
        }
    std::cout<<"";
}//blank 0
void CLayer::initWeights(){
    for(int i = 0; i < this->tensors.size(); i ++){
        float randBias = (rand() % 1000)/100.f;
        tensors[i].second = randBias;
        for(int z = 0; z < tensors[i].first.matrices.size(); z ++){
            for(int y = 0; y < tensors[i].first.matrices[z].weights.size(); y ++){
                for(int x = 0; x < tensors[i].first.matrices[z].weights[y].size(); x ++){
                    float randWeight = (rand() % 100)/100.f;
                    tensors[i].first.matrices[z].weights[y][x] = randWeight;
                }
            }
        }
    }
}
void CLayer::run(const Tensor &inputTensor){
    if(inputTensor.matrices.size() != this->tensors[0].first.matrices.size()){
        throw std::invalid_argument( "tensor dimensions wont match!\n" );
        return;
    }

    
    for(auto k : this->tensors){
        Matrix result = Functions::convolve(inputTensor, k.first, this->stride, this->padding);//add bias
        outputTensor->matrices.push_back(result);
    }
    

    
}
CLayer::~CLayer(){}
