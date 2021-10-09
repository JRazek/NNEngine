//
// Created by j.razek on 07.10.2021.
//

#ifndef NEURALFLOWS_RECURRENTOUTPUTLAYER_H
#define NEURALFLOWS_RECURRENTOUTPUTLAYER_H
#include "../../interfaces/Layer.h"
#include "../RecurrentLayer.h"

namespace cn {
    class RecurrentOutputLayer : public Layer{
        RecurrentLayer *parentLayer;

        void CPURun(const Tensor<double> &_input) override;
        virtual JSON jsonEncode() const override;
        virtual double getChain(const Vector4<int> &inputPos) override;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;

    public:
        explicit RecurrentOutputLayer(const Vector3<int> &_inputSize, RecurrentLayer &_parentLayer);
    };
}

#endif //NEURALFLOWS_RECURRENTOUTPUTLAYER_H
