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
        const DifferentiableFunction &differentiableFunction;
        std::optional<Bitmap<float>> netSums;

    public:
        void randomInit() override;
        /**
         *
         * @param _id __id of layer in network
         * @param _network network to which layer belongs
         * @param _differentiableFunction function with its derivative
         * @param _neuronsCount input size (neuron count)
         */
        FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network);

        Bitmap<float> run(const Bitmap<float> &input) override;

        virtual float getChain(const Vector3<int> &inputPos) override;

        float diffWeight(int weightID);
        std::vector<float> getGradient() override;

        int weightsCount() const override;

        void setWeight(int weightID, float value) override;
        float getWeight(int weightID) const override;

    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
