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
        float learningRate;
        std::vector<std::vector<float>> memorizedWeights;
        std::vector<std::vector<float>> memorizedBiases;
    public:
        Backpropagation(Network &_network, float _learningRate, int _miniBatchSize);
        void propagate(const Bitmap<float> &target);
        float getError(const cn::Bitmap<float> &target) const;
    };

}
#endif //NEURALNETLIBRARY_BACKPROPAGATION_H
