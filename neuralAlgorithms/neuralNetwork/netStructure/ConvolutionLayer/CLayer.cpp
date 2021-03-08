#include "CLayer.h"
#include <utils/Functions.h>
#include <activations/SigmoidFunction.h>
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
CLayer::CLayer(int id, Net * net, const CLayer &p1, const CLayer &p2, float mutationRate, int seed):
    kernelSizeX(p1.tensors[0].first.getX()), kernelSizeY(p1.tensors[0].first.getY()), kernelSizeZ(p1.tensors[0].first.getZ()),
    stride(1), padding(0), outputTensor(), activationFunction(activationFunction),
    Layer(id, net){

    srand(seed);
    for(int i = 0; i < p1.tensors.size(); i ++){
        std::pair<const Tensor *, float> t1 = {&p1.tensors[i].first, p1.tensors[i].second};
        std::pair<const Tensor *, float> t2 = {&p2.tensors[i].first, p2.tensors[i].second};
        if(t1.first->getZ() != t2.first->getZ()){
            throw std::invalid_argument( "tensor dimensions wont match! cannot cross!!\n" );
            return; 
        }
        Tensor childTensor = Tensor(t1.first->getX(), t1.first->getY(), t1.first->getZ());
        float bias = (rand() % 2) ? t1.second : t2.second;
        //this->activationFunction = new ActivationFunction();
        //clone!!todotodo!!!!
        this->activationFunction = new SigmoidFunction();
        /////////////////////
        for(int z = 0; z < t1.first->getZ(); z ++){
            //mix the tensor
            for(int y = 0; y < t1.first->getY(); y++){
                for(int x = 0; x < t1.first->getX(); x++){
                    float val = (rand() % 2) ? t1.first->getValue(x, y, z) : t2.first->getValue(x, y, z);
                    childTensor.edit(x, y, z, val);
                }
            }
        }
        tensors.push_back({childTensor, bias});
    }
    if(rand() % 1000 <= mutationRate * 1000){
        int tensorID = rand() % this->tensors.size();/*
        for(auto w : this->tensors[tensorID]->inputEdges){
            w.second = ((float)(rand() % 10000))/10000;
            edge.second *= ((rand() % 2) ? -1 : 1);
        }*/
        for(int z = 0; z < tensors[tensorID].first.getZ(); z++){
            for(int y = 0; y < tensors[tensorID].first.getY(); y++){
                for(int x = 0; x < tensors[tensorID].first.getX(); x++){
                    tensors[tensorID].first.edit(x, y, z, ((float)(rand() % 10000))/10000 * ((rand() % 2) ? -1 : 1));
                }
            }
        }
        this->tensors[tensorID].second = ((float)(rand() % 10000))/10000;
        this->tensors[tensorID].second *= ((rand() % 2) ? -1 : 1);
    }
}
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
