//
// Created by user on 06.09.2021.
//

#ifndef NEURALNETLIBRARY_ACTIVATIONLAYER_H
#define NEURALNETLIBRARY_ACTIVATIONLAYER_H
#include "Layer.h"

namespace cn {
    class ActivationLayer : public Layer {
        ActivationLayer(int id, Network &network);
    };
}

#endif //NEURALNETLIBRARY_ACTIVATIONLAYER_H
