#include "FFLayer.h"
#include <netStructure/ConvolutionLayer/CLayer.h>
#include <Net.h>
#include <cstdlib>
#include <iostream>

FFLayer::FFLayer(int id, Net * net, int neuronsCount, ActivationFunction * f):Layer(id, net, neuronsCount){
    this->activationFunction = f;
    for(int i = 0; i < neuronsCount; i ++){
        this->neurons.push_back(new Neuron(i));
    }
}
FFLayer::Neuron::Neuron(int idInLayer):idInLayer(idInLayer){}
void FFLayer::initConnections(int seed = 0){
    srand (seed);
    Layer * prevLayer = Layer::net->layers[Layer::idInNet - 1];
    for(auto n : this->neurons){
        float randBias = (rand() % 1000)/100.f;
        int inputSize = prevLayer->outputVectorSize;
        for(int i = 0; i < inputSize; i ++){
            float randWeight = (rand() % 100)/100.f;
            n->inputEdges.push_back({i, randWeight});
        }
    }
}

FFLayer::~FFLayer(){
    for(auto n : this->neurons){
        delete n;
    }
}
void FFLayer::run(const std::vector<float> &input){
    std::cout<<"Im a fucking FFLayer!\n";
}