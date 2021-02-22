#pragma once
#include <Layer.h>
#include <vector>

struct FFLayer : Layer{
    std::vector<Neuron *> neurons;
};