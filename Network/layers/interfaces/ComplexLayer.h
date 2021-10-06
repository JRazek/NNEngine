//
// Created by user on 06.10.2021.
//

#ifndef NEURALFLOWS_COMPLEXLAYER_H
#define NEURALFLOWS_COMPLEXLAYER_H

#include "Layer.h"

namespace cn {
    class ComplexLayer : public Layer{
    public:
        virtual void ready() = 0;
        ComplexLayer(Vector3<int> _inputSize);
        ComplexLayer(const Layer &layer);
        ComplexLayer(Layer &&layer);
    };
}

#endif //NEURALFLOWS_COMPLEXLAYER_H
