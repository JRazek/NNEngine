#include "FFLayer.h"

FFLayer::FFLayer(int id, int neuronsCount):Layer(id){
    
}
FFLayer::Neuron::Neuron(int idInLayer):idInLayer(idInLayer){}
FFLayer::~FFLayer(){
    for(auto n : this->neurons){
        delete n;
    }
}