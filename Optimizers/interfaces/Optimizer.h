//
// Created by user on 05.09.2021.
//

#ifndef NEURALNETLIBRARY_OPTIMIZER_H
#define NEURALNETLIBRARY_OPTIMIZER_H

#include "../../Network/Network.h"

namespace cn {
    class Optimizer {
    protected:
        //todo change to pointer
        Network &network;
        int iteration;
        double learningRate;
        Optimizer(Network &_network, double _learningRate);
        const std::vector<cn::Layer *> &getNetworkLayers() const;
        const std::vector<cn::Learnable *> &getLearnables() const;

        //until network changed for pointer
        Optimizer(const Optimizer &optimizer) = delete;
        Optimizer &operator=(const Optimizer &optimizer) = delete;

    public:
        virtual void propagate(const Bitmap<double> &target, bool CUDAAccelerate = false) = 0;
        double getError(const cn::Bitmap<double> &target) const;
    };
}


#endif //NEURALNETLIBRARY_OPTIMIZER_H
