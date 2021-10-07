//
// Created by j.razek on 07.10.2021.
//

#ifndef NEURALFLOWS_RECURRENTOUTPUTLAYER_H
#define NEURALFLOWS_RECURRENTOUTPUTLAYER_H
#include "../../interfaces/Layer.h"
#include "../RecurrentLayer.h"

namespace cn {
    class RecurrentOutputLayer : public Layer{
        RecurrentLayer *parentLayer;
        explicit RecurrentOutputLayer(const Vector3<int> &_inputSize, RecurrentLayer &parent);
    };
}

#endif //NEURALFLOWS_RECURRENTOUTPUTLAYER_H
