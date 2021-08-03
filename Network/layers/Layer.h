//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_LAYER_H
#define NEURALNETLIBRARY_LAYER_H

#include <vector>
#include "../Network.h"
typedef unsigned char byte;

class Layer {
protected:
    Network * network;
public:
    const int id;
    Layer(int id, Network * network);
    virtual void run() = 0;

    virtual ~Layer();
};


#endif //NEURALNETLIBRARY_LAYER_H
