//
// Created by user on 21.08.2021.
//

#ifndef NEURALNETLIBRARY_BACKPROPAGATION_H
#define NEURALNETLIBRARY_BACKPROPAGATION_H

namespace cn {
    template<typename T>
    struct Bitmap;
    class Network;

    class Backpropagation {

        Network &network;
        int iteration;

        Backpropagation(Network &_network);
        void propagate(const Bitmap<float> &target);
    };

}
#endif //NEURALNETLIBRARY_BACKPROPAGATION_H
