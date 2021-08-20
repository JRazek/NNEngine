//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include "interfaces/Layer.h"
#include "../../Utils/Differentiables/DifferentiableFunction.h"
#include "interfaces/RandomInitiable.h"

namespace cn {
    class Network;

    class FFLayer : public cn::Layer, public RandomInitiable{
        std::vector<float> biases;
        std::vector<float> weights;
        const DifferentiableFunction &differentiableFunction;

        const int neuronsCount;
    public:
        void randomInit() override ;
        /**
         *
         * @param _id id of layer in network
         * @param _network network to which layer belongs
         * @param _differentiableFunction function with its derivative
         * @param _neuronsCount input size (neuron count)
         */
        FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network *_network);

        void run(const Bitmap<float> &bitmap) override;
        /*
         * returns ith weight belonging to the neuron
         */
        float getWeight(int neuron, int weightID);
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
