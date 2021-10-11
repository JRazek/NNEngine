//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_FFLAYER_H
#define NEURALNETLIBRARY_FFLAYER_H

#include <random>
#include "../interfaces/Layer.h"
#include "../interfaces/Learnable.h"

namespace cn {
    class Network;

    class FFLayer : public Learnable{
        std::vector<double> biases;
        std::vector<double> weights;

        int neuronsCount;

    protected:


    public:
        /**
         *
         * @param _network network to which layer belongs
         * @param _differentiableFunction function with its derivative
         * @param _neuronsCount input size (neuron count)
         */
        FFLayer(Vector3<int> _inputSize, int _neuronsCount);

        virtual std::vector<double *> getWeightsByRef() override;
        virtual std::vector<double *> getBiasesByRef() override;
        FFLayer(const JSON &json);

        void CPURun(const Tensor<double> &_input) override;

        virtual double getChain(const Vector4<int> &inputPos) override;

        void randomInit(std::default_random_engine &randomEngine) override;
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

        std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;
    };
}


#endif //NEURALNETLIBRARY_FFLAYER_H
