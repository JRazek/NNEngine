//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H
#include "Layer.h"

namespace cn {
    class Network;
    class Learnable : public Layer{
    public:
        virtual void randomInit() = 0;
        Learnable(int id, Network &network);
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
