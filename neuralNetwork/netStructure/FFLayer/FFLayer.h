#pragma once
#include "../Layer/Layer.h"
#include <vector>
#include <utility>

struct FFLayer : Layer{
    FFLayer(int id, int neuronsCount);
    struct Neuron{
        const int idInLayer;
        std::vector< std::pair<int, float> > inputEdges; //id in prev layer, weights
        float bias;
        Neuron(int idInLayer);
    };
    std::vector<Neuron *> neurons;
    ~FFLayer();
};