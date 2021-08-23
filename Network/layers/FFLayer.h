//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include "interfaces/Layer.h"
#include "../../Utils/Differentiables/DifferentiableFunction.h"
#include "interfaces/Learnable.h"

namespace cn {
    class Network;

    class FFLayer : public Learnable{
        std::vector<float> biases;
        std::vector<float> weights;

    public:
        void randomInit() override ;
        /**
         *
         * @param _id id of layer in network
         * @param _network network to which layer belongs
         * @param _differentiableFunction function with its derivative
         * @param _neuronsCount input size (neuron count)
         */
        FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network);

        void run(const Bitmap<float> &bitmap) override;

        /**
         * returns ith weight belonging to the neuron
         */
        float getWeight(int neuron, int weightID);

        float getChain(int neuronID) override;
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
