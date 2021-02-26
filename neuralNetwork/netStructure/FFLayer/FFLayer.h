#pragma once
#include <netStructure/Layer/Layer.h>
#include <vector>
#include <utility>

struct FFLayer : Layer{
    FFLayer(int id, Net * net, int neuronsCount);
    struct Neuron{
        const int idInLayer;
        std::vector< std::pair<int, float> > inputEdges; //id in prev layer, weights  //if empty - first layer
        float bias;
        Neuron(int idInLayer);
    };
    //virtual void run() override;
    void initConnections(int seed);
    std::vector<Neuron *> neurons;
    ~FFLayer();
};