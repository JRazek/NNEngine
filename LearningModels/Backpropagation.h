//
// Created by user on 21.08.2021.
//

#ifndef NEURALNETLIBRARY_BACKPROPAGATION_H
#define NEURALNETLIBRARY_BACKPROPAGATION_H

namespace cn {
    class Network;

    class Backpropagation {

        Network *network;

        Backpropagation(Network &_network);
    };

}
#endif //NEURALNETLIBRARY_BACKPROPAGATION_H
