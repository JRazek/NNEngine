//
// Created by user on 05.09.2021.
//

#ifndef NEURALNETLIBRARY_MOMENTUMGD_H
#define NEURALNETLIBRARY_MOMENTUMGD_H

#include "../Network/Network.h"

namespace cn {
    class MomentumGD {
        Network &network;
        int samplesCount;
        double learningRate;
        MomentumGD(Network &_network, int _samplesCount, double _learningRate);
    };
}

#endif //NEURALNETLIBRARY_MOMENTUMGD_H
