//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include "interfaces/Layer.h"
#include "interfaces/Learnable.h"

namespace cn {
    class Network;

    class FFLayer : public Learnable{
        std::vector<double> biases;
        std::vector<double> weights;

    public:
        void randomInit() override;
        /**
         *
         * @param _id __id of layer in network
         * @param _network network to which layer belongs
         * @param _differentiableFunction function with its derivative
         * @param _neuronsCount input size (neuron count)
         */
        FFLayer(int _id, int _neuronsCount, Network &_network);

        FFLayer(Network &_network, const JSON &json);

        Bitmap<double> run(const Bitmap<double> &input) override;

        virtual double getChain(const Vector3<int> &inputPos) override;

        double diffWeight(int weightID);
        double diffBias(int neuronID);
        std::vector<double> getWeightsGradient() override;
        std::vector<double> getBiasesGradient() override;


        void setBias(int neuronID, double value) override;
        double getBias(int neuronID) const override;

        int weightsCount() const override;
        int biasesCount() const override;

        void setWeight(int weightID, double value) override;
        double getWeight(int weightID) const override;

        JSON jsonEncode() const override;
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
