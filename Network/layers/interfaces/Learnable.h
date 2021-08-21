//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H
#include "Layer.h"

namespace cn {
    struct Learnable : public Layer{

        const int neuronsCount;
        Learnable(int id, cn::Network &network, int _neuronsCount);

        /**
         * dp for chain and computation saving
         */
        float memoizedChain;
        /**
         * info if chain is saved
         */
        bool memoizedChainFlag = false;

        /**
         * randomly initializes all the learnable elements in the layer
         */
        virtual void randomInit() = 0;
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
