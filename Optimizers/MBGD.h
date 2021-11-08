//
// Created by user on 21.08.2021.
//

#ifndef NEURALNETLIBRARY_MBGD_H
#define NEURALNETLIBRARY_MBGD_H
#include <vector>
#include "interfaces/GradientOptimizer.h"
namespace cn {
    template<typename T>
    class Tensor;
    class Network;

    class MBGD : public GradientOptimizer{
        int miniBatchSize;
        std::vector<std::vector<double>> memorizedWeights;
        std::vector<std::vector<double>> memorizedBiases;
    public:
        MBGD(Network &_network, double _learningRate, int _miniBatchSize);
        void propagate(const Tensor<double> &target) override;
    };

}
#endif //NEURALNETLIBRARY_MBGD_H
