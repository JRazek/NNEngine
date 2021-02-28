#include "CLayer.h"
#include <utils/Functions.h>
#include <Net.h>
#include <iostream>
//todo
CLayer::CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY, ActivationFunction * activationFunction):
    kernelSizeX(matrixSizeX), kernelSizeY(matrixSizeY), kernelSizeZ(tensorDepth), stride(1), padding(0), outputTensor(tensorCount), activationFunction(activationFunction),
    Layer(id, net, tensorCount * matrixSizeX * matrixSizeY * tensorDepth){
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

    
    for(auto k : this->tensors){
        if(inputTensor.z != k.first.z){
            throw std::invalid_argument( "tensor dimensions wont match!\n" );
            return;
        }

        Matrix result = Functions::convolve(inputTensor, k.first, this->padding, this->stride);//add bias
        for(int y = 0; y < result.y; y ++){
            for(int x = 0; x < result.x; x++){
                result.weights[x][y] = this->activationFunction->getValue(result.weights[x][y] + k.second);
            }
        }
        outputTensor.matrices.push_back(result);
        std::vector<float> flattened;
        for(int y = 0; y < result.y; y++){
            for(int x = 0; x < result.x; x ++){
                flattened.push_back(result.weights[x][y]);
            }
        }
        this->outputVector.insert(outputVector.end(), flattened.begin(), flattened.end());
    }
    

    
}
CLayer::~CLayer(){
    delete activationFunction;
}
