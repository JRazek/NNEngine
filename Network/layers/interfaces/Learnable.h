//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H
#include "Layer.h"

namespace cn {
    class Learnable : public Layer{
    public:

        const int neuronsCount;
        Learnable(int id, cn::Network &network, int _neuronsCount,
                  const DifferentiableFunction &differentiableFunction);

        std::optional<Bitmap<float>> netValues;

        const DifferentiableFunction &activationFunction;

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

        virtual float diffWeight(int neuronID, int weightID) = 0;

        virtual float &getWeight(int neuronID, int weightID) = 0;

        virtual void randomInit() = 0;
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
