//
// Created by user on 05.09.2021.
//

#ifndef NEURALNETLIBRARY_MOMENTUMGD_H
#define NEURALNETLIBRARY_MOMENTUMGD_H

#include "interfaces/GradientOptimizer.h"
namespace cn {
    class MomentumGD : public GradientOptimizer{
        float theta;
        std::vector<std::vector<float>> emaWeightsMemo;
        std::vector<std::vector<float>> emaBiasesMemo;
    public:
        MomentumGD(Network &_network, float _theta, double _learningRate);
        void propagate(const Tensor<double> &target) override;
    };
}

#endif //NEURALNETLIBRARY_MOMENTUMGD_H
