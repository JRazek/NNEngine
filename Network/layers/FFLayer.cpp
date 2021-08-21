//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"

cn::FFLayer::FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network) :
        cn::Layer(_id, _network),
        differentiableFunction(_differentiableFunction),
        biases(_neuronsCount),
        Learnable(_neuronsCount){
    if(id == 0){
        throw std::logic_error("FFLayer must not be the first layer in the network!");
    }else{
        Bitmap<float> *prev = &network->layers[id - 1]->output.value();
        if(prev->w < 1 || prev->h != 1 || prev->d != 1){
            throw std::logic_error("There must be a vector output layer before FFLayer!");
        }
        weights = std::vector<float>(neuronsCount * prev->w);
    }
    output.emplace(Bitmap<float>(neuronsCount, 1, 1));
}

void cn::FFLayer::run(const Bitmap<float> &bitmap) {
    if(bitmap.h != 1 || bitmap.d != 1 || bitmap.w < 1){
        throw std::logic_error("bitmap input to ff layer must be a normalized vector type!");
    }
    Bitmap<float> *input = &network->layers[id - 1]->output.value();

    for(int n = 0; n < neuronsCount; n ++){
        float sum = biases[n];
        for(int i = 0; i < input->w; i ++){
            sum += getWeight(n, i) * input->getCell(i, 0, 0);
        }
        output->setCell(n, 0, 0, differentiableFunction.func(sum));
    }
}

void cn::FFLayer::randomInit() {
    for(auto &w : weights){
        w = network->getRandom(-1, 1);
    }
    for(auto &b : biases){
        b = network->getRandom(-5, 5);
    }
}

float cn::FFLayer::getWeight(int neuron, int weightID) {
    int perNeuron = weights.size() / neuronsCount;
    return weights[perNeuron * neuron + weightID];
}

float cn::FFLayer::getChain(int neuronID) {
    return 0;
}
