#pragma once
#include "../Layer/Layer.h"
#include <vector>

struct FFLayer : Layer{
    struct Neuron{
        const int id;
        std::vector<float> inputEdges;
        float bias;
        Neuron(int id);
    };
    std::vector<Neuron *> neurons;    
    FFLayer(int id, int neuronsCount);
};