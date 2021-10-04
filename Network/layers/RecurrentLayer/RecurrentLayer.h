//
// Created by jrazek on 24.09.2021.
//

#ifndef NEURALNETLIBRARY_RECURRENTLAYER_H
#define NEURALNETLIBRARY_RECURRENTLAYER_H

#include <stack>
#include "../interfaces/Layer.h"

namespace cn {
    class RecurrentLayer : public Layer {
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
        void CPURun(const Tensor<double> &_input) override;
        double getChain(const Vector4<int> &inputPos) override;
        RecurrentLayer(const JSON &json);
        Tensor<double> identity;

        JSON jsonEncode() const override;

    public:
        RecurrentLayer(cn::Vector3<int> _inputSize);
    };
}

#endif //NEURALNETLIBRARY_RECURRENTLAYER_H
