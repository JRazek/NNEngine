//
// Created by jrazek on 24.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H
#include "Layer.h"

namespace cn {
    class Network;
    class Learnable : public Layer{
    protected:
        const Bitmap<float> *_input;
    public:
        virtual void randomInit() = 0;
        Learnable(int id, Network &network);
        virtual float diffWeight(int neuronID, int weightID) = 0;
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
