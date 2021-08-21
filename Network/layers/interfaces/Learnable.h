//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H

namespace cn {
    struct Learnable {

        const int neuronsCount;
        Learnable(int _neuronsCount);

        /**
         * randomly initializes all the learnable elements in the layer
         */
        virtual void randomInit() = 0;

        /**
         *
         * @param neuronID - in case of FF - neuron, in case of Conv - kernel
         * @return differentiation chain calculated from that neuron
         */
        virtual float getChain(int neuronID) = 0;
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
