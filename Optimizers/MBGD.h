//
// Created by user on 21.08.2021.
//

#ifndef NEURALNETLIBRARY_MBGD_H
#define NEURALNETLIBRARY_MBGD_H
#include <vector>
namespace cn {
    template<typename T>
    class Bitmap;
    class Network;

    class MBGD {

        Network &network;
        int iteration;
        int miniBatchSize;
        double learningRate;
        std::vector<std::vector<double>> memorizedWeights;
        std::vector<std::vector<double>> memorizedBiases;
    public:
        MBGD(Network &_network, double _learningRate, int _miniBatchSize);
        void propagate(const Bitmap<double> &target);
        double getError(const cn::Bitmap<double> &target) const;
    };

}
#endif //NEURALNETLIBRARY_MBGD_H
