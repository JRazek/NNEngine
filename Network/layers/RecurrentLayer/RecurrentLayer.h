//
// Created by jrazek on 24.09.2021.
//

#ifndef NEURALNETLIBRARY_RECURRENTLAYER_H
#define NEURALNETLIBRARY_RECURRENTLAYER_H

#include <stack>
#include "../interfaces/Learnable.h"

namespace cn {
    class RecurrentOutputLayer;
    class RecurrentLayer : public Learnable {
        friend RecurrentOutputLayer;
        std::vector<std::unique_ptr<Layer>> internalLayers;
        std::vector<Learnable *> learnableLayers;
        std::vector<double *> weights;
        std::vector<double *> biases;

        void CPURun(const Tensor<double> &_input) override;
        double getChain(const Vector4<int> &inputPos) override;
        double getChainFromChild(const Vector4<int> &inputPos);
        void appendLearnable(Learnable *learnable);
        Tensor<double> identity;
        bool _ready = false;
    protected:
        virtual std::vector<double *> getWeightsByRef() override;
        virtual std::vector<double *> getBiasesByRef() override;

    public:
        explicit RecurrentLayer(const JSON &json);
        explicit RecurrentLayer(const Vector3<int> &_inputSize);

        RecurrentLayer(const RecurrentLayer &recurrentLayer);
        RecurrentLayer(RecurrentLayer &&recurrentLayer) = default;

        void randomInit(std::default_random_engine &randomEngine) override;

        std::vector<double> getWeightsGradient() override;
        std::vector<double> getBiasesGradient() override;


        void setBias(int neuronID, double value) override;
        double getBias(int neuronID) const override;

        int weightsCount() const override;
        int biasesCount() const override;

        void setWeight(int weightID, double value) override;
        double getWeight(int weightID) const override;

        void appendFFLayer(int neuronsCount);
        void appendSigmoidLayer();

        std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;

        JSON jsonEncode() const override;
        void ready();
    };
}

#endif //NEURALNETLIBRARY_RECURRENTLAYER_H
