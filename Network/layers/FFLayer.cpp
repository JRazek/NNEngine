//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"

cn::FFLayer::FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network) :
        biases(_neuronsCount),
        Learnable(_id, _network, _neuronsCount, _differentiableFunction) {
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
    Bitmap<float> &input = network->layers[id - 1]->output.value();

    netValues.emplace(input.w, input.h, input.d);
    //std::copy(input.data(), input.data() + input.w * input.h * input.d, netValues.value().data());

    for(int n = 0; n < neuronsCount; n ++){
        float sum = biases[n];
        for(int i = 0; i < input.w; i ++){
            sum += getWeight(n, i) * input.getCell(i, 0, 0);
        }
        netValues->setCell(n, 0, 0, sum);
        output->setCell(n, 0, 0, activationFunction.func(sum));
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

float &cn::FFLayer::getWeight(int neuron, int weightID) {
    int perNeuron = weights.size() / neuronsCount;
    return weights[perNeuron * neuron + weightID];
}

float cn::FFLayer::getChain(int neuronID) {
    return activationFunction.derive(netValues->getCell(neuronID, 0, 0)) * [this](){
      float res = 1;
      if(id != network->layers.size() - 1){
          float tmp = 0;
          for(int i = 0; i < output->w; i ++){
              tmp += network->layers[id + 1]->getChain(i);
          }
      }
      return res;
    }();
}

float cn::FFLayer::diffWeight(int neuronID, int weightID) {
    Bitmap<float> *prevInput = (id != 0) ? &network->layers[id - 1]->output.value() : &network->input.value();
    weightID = neuronID * (neuronsCount/weights.size()) + weightID;
    return output->getCell(weightID, 0, 0) * getChain(neuronID);
}
