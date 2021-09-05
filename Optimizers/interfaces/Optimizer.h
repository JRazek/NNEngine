//
// Created by user on 05.09.2021.
//

#ifndef NEURALNETLIBRARY_OPTIMIZER_H
#define NEURALNETLIBRARY_OPTIMIZER_H

#include "../../Network/Network.h"

namespace cn {
    class Optimizer {
    protected:
        Network &network;
        int iteration;
        double learningRate;
        Optimizer(Network &_network, double _learningRate);

    public:
        virtual void propagate(const Bitmap<double> &target) = 0;
        double getError(const cn::Bitmap<double> &target) const;
    };
}


#endif //NEURALNETLIBRARY_OPTIMIZER_H
