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

    FFLayer(int id, Net * net, int inputVectorSize, int neuronsCount, ActivationFunction * f);
    FFLayer(int id, Net * net, const FFLayer &p1, const FFLayer &p2, int seed);
    const int inputVectorSize;
    ActivationFunction * activationFunction;
    void initConnections();
    void run(const std::vector<float> &input);
    std::vector<Neuron *> neurons;
    ~FFLayer();
};