#include "CLayer.h"
#include <utils/Functions.h>
#include <Net.h>
#include <iostream>
//todo
CLayer::CLayer(int id, Net * net, int tensorCount, int matrixSizeX, int matrixSizeY, int tensorDepth, ActivationFunction * activationFunction):
    kernelSizeX(matrixSizeX), kernelSizeY(matrixSizeY), kernelSizeZ(tensorDepth), stride(1), padding(0), outputTensor(), activationFunction(activationFunction),
    Layer(id, net){
        if(tensorDepth == 4)
            std::cout<<"";
        for(int i = 0; i < tensorCount; i ++){
            Tensor tensor = Tensor(matrixSizeX, matrixSizeY, tensorDepth);
            this->tensors.push_back({tensor, 0});
        }
    std::cout<<"";
}//blank 0
void CLayer::initWeights(){
    for(int i = 0; i < this->tensors.size(); i ++){
        float randBias = (rand() % 1000)/100.f * (rand() % 2 == 1 ? 1 : -1);
        tensors[i].second = randBias;
        for(int z = 0; z < tensors[i].first.getZ(); z ++){
            for(int y = 0; y < tensors[i].first.getY(); y ++){
                for(int x = 0; x < tensors[i].first.getX(); x ++){
                    float randWeight = (rand() % 100)/100.f * (rand() % 2 == 1 ? 1 : -1);
                    tensors[i].first.edit(x, y, z, randWeight);
                }
            }
        }
    }
}
void CLayer::run(const Tensor &inputTensor){
    this->outputTensor.clearMatrices();
    this->outputVector.clear();
    for(auto k : this->tensors){
        if(inputTensor.getZ() != k.first.getZ()){
            throw std::invalid_argument( "tensor dimensions wont match!\n" );
            return;
        }

        Matrix result = Functions::convolve(inputTensor, k.first, this->padding, this->stride);//add bias
        for(int y = 0; y < result.getY(); y ++){
            for(int x = 0; x < result.getX(); x++){
                float newVal = this->activationFunction->getValue(result.getValue(x, y) + k.second);
                result.edit(x, y, newVal);
            }
        }
        outputTensor.pushMatrix(result);
        std::vector<float> flattened;
        for(int y = 0; y < result.getY(); y++){
            for(int x = 0; x < result.getX(); x ++){
                flattened.push_back(result.getValue(x, y));
            }
        }
        this->outputVector.insert(outputVector.end(), flattened.begin(), flattened.end());
    }
    

    
}
CLayer::~CLayer(){
    delete activationFunction;
}
