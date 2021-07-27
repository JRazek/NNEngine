//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_LAYER_H
#define NEURALNETLIBRARY_LAYER_H

#include <vector>
#include "../Network.h"
typedef unsigned char byte;

class Layer {
private:
    Network * network;
public:
    Layer(Network * network);
    virtual void run() = 0;
    virtual std::vector<float> getOutput() = 0;
};


#endif //NEURALNETLIBRARY_LAYER_H
