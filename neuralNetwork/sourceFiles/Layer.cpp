#pragma once

#include <Layer.h>

Layer::Layer(int id, int neuronsCount):id(id){
    this->neurons.reserve(neuronsCount);
    for(int i = 0; i < neuronsCount; i ++){
        neurons.push_back(new Neuron(i));
    }
}