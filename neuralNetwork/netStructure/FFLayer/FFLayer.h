#pragma once
#include <netStructure/Layer/Layer.h>
#include <vector>
#include <utility>
#include <activations/ActivationFunction.h>

struct FFLayer : Layer{
    struct Neuron{
        const int idInLayer;
        std::vector< std::pair<int, float> > inputEdges; //id in prev layer, weights  //if empty - first layer
        float bias;
        Neuron(int idInLayer);
    };

    FFLayer(int id, Net * net, int neuronsCount, ActivationFunction * f);
    ActivationFunction * activationFunction;
    void initConnections(int seed);
    void run(const std::vector<float> &input) override;
    std::vector<Neuron *> neurons;
    ~FFLayer();
};