#include "FFLayer.h"
#include <netStructure/ConvolutionLayer/CLayer.h>
#include <Net.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

FFLayer::FFLayer(int id, Net * net, int inputVectorSize, int neuronsCount, ActivationFunction * f):
    inputVectorSize(inputVectorSize), activationFunction(f), 
    Layer(id, net){
        for(int i = 0; i < neuronsCount; i ++){
            this->neurons.push_back(new Neuron(i));
        }
}


FFLayer::FFLayer(int id, Net * net, const FFLayer &p1, const FFLayer &p2, int seed):
    inputVectorSize(p1.inputVectorSize), activationFunction(p1.activationFunction), 
        Layer(id, net){
    if(p1.neurons.size() != p2.neurons.size()){
        throw std::invalid_argument( "cannot cross two different layers!" );
        return;
    }
    srand(seed);
    for(int i = 0; i < p1.neurons.size(); i ++){
        FFLayer::Neuron * n1 = p1.neurons[i];
        FFLayer::Neuron * n2 = p2.neurons[i];

        FFLayer::Neuron * child = new Neuron(i);

        for(int j = 0; j < n1->inputEdges.size(); j++){
            bool r = rand() % 2;
            std::pair<int, float> edge = r ? n1->inputEdges[j] : n2->inputEdges[j];
            child->inputEdges.push_back(edge);
        }
        this->neurons.push_back(child);
    }
}

FFLayer::Neuron::Neuron(int idInLayer):idInLayer(idInLayer){}
void FFLayer::initConnections(){
    Layer * prevLayer = Layer::net->layers[Layer::idInNet - 1];
    for(auto n : this->neurons){
        float randBias = (rand() % 1000)/100.f * (rand() % 2 == 1 ? 1 : -1);
        int inputSize = this->inputVectorSize;
        for(int i = 0; i < inputSize; i ++){
            float randWeight = (rand() % 100)/100.f * (rand() % 2 == 1 ? 1 : -1);
            n->inputEdges.push_back({i, randWeight});
        }
    }
}

FFLayer::~FFLayer(){
    for(auto n : this->neurons){
        delete n;
    }
    delete activationFunction;
}

void FFLayer::run(const std::vector<float> &input){
    if(this->neurons.size() <= 0){
        throw std::invalid_argument( "neurons not initiated in layer " + this->idInNet );
        return;
    }

    if(input.size() != this->neurons[0]->inputEdges.size()){
        throw std::invalid_argument( "wrong input size!" );
        return;
    }
    this->outputVector.clear();

    for(int i = 0; i < neurons.size(); i ++){
        Neuron * n = neurons[i];
        float sum = 0;
        for(int j = 0; j < n->inputEdges.size(); j ++){
            sum += input[j] * n->inputEdges[j].second;
        }
        sum += n->bias;
        
        float activatedVal = activationFunction->getValue(sum);
        this->outputVector.push_back(activatedVal);
    }
}