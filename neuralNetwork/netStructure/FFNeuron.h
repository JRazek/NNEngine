#pragma once

#include <vector>

struct Neuron{
    const int id;
    float bias;
    std::vector<Neuron *> inputEdges; 
    Neuron(int id);
};