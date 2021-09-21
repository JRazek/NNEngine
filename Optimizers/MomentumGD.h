//
// Created by user on 05.09.2021.
//

#ifndef NEURALNETLIBRARY_MOMENTUMGD_H
#define NEURALNETLIBRARY_MOMENTUMGD_H

#include "interfaces/Optimizer.h"
namespace cn {
    class MomentumGD : public Optimizer{
        float theta;
        std::vector<std::vector<float>> emaWeightsMemo;
        std::vector<std::vector<float>> emaBiasesMemo;
    public:
        MomentumGD(Network &_network, float _theta, double _learningRate);
        void propagate(const Bitmap<double> &target, bool CUDAAccelerate) override;
    };
}

#endif //NEURALNETLIBRARY_MOMENTUMGD_H
