//
// Created by user on 21.08.2021.
//

#ifndef NEURALNETLIBRARY_BACKPROPAGATION_H
#define NEURALNETLIBRARY_BACKPROPAGATION_H
#include <vector>
namespace cn {
    template<typename T>
    struct Bitmap;
    class Network;

    class Backpropagation {

        Network &network;
        int iteration;
        int miniBatchSize;
        double learningRate;
        std::vector<std::vector<double>> memorizedWeights;
        std::vector<std::vector<double>> memorizedBiases;
    public:
        Backpropagation(Network &_network, double _learningRate, int _miniBatchSize);
        void propagate(const Bitmap<double> &target);
        double getError(const cn::Bitmap<double> &target) const;
    };

}
#endif //NEURALNETLIBRARY_BACKPROPAGATION_H
