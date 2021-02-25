#include "FFLayer.h"
#include <Net.h>
#include <cstdlib>

FFLayer::FFLayer(int id, Net * net, int neuronsCount):Layer(id, net){
    
}
FFLayer::Neuron::Neuron(int idInLayer):idInLayer(idInLayer){}
void FFLayer::initConnections(int seed = 0){
    srand (seed);
    Layer * prevLayer = Layer::net->layers[Layer::idInNet - 1];

    if(dynamic_cast<FFLayer *>(prevLayer) != nullptr){
        //it is a FFLayer
    }
    if(dynamic_cast<CLayer *>(prevLayer) != nullptr){
        //it is a CLayer
    }
    float randWeight = (rand() % 100)/100.f;
    float randBias = (rand() % 1000)/100.f;
}
FFLayer::~FFLayer(){
    for(auto n : this->neurons){
        delete n;
    }
}