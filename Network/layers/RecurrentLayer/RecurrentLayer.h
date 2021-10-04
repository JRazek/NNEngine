//
// Created by jrazek on 24.09.2021.
//

#ifndef NEURALNETLIBRARY_RECURRENTLAYER_H
#define NEURALNETLIBRARY_RECURRENTLAYER_H

#include <stack>
#include "../interfaces/Layer.h"

namespace cn {
    class RecurrentLayer : public Layer {
        std::vector<std::unique_ptr<Layer>> internalLayers;
        void CPURun(const Tensor<double> &_input) override;
        double getChain(const Vector4<int> &inputPos) override;
        Tensor<double> identity;


        JSON jsonEncode() const override;
        RecurrentLayer(cn::Vector3<int> _inputSize);

    public:
        RecurrentLayer(const JSON &json);
        RecurrentLayer(cn::Vector3<int> _inputSize, std::vector<std::unique_ptr<Layer>> &&layers);
        RecurrentLayer(const RecurrentLayer &recurrentLayer);
        RecurrentLayer(RecurrentLayer &&recurrentLayer) = default;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;
    };
}

#endif //NEURALNETLIBRARY_RECURRENTLAYER_H
