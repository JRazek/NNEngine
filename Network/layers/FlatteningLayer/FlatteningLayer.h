//
// Created by jrazek on 19.08.2021.
//

#ifndef NEURALNETLIBRARY_FLATTENINGLAYER_H
#define NEURALNETLIBRARY_FLATTENINGLAYER_H

#include "../interfaces/Layer.h"

namespace cn {
    class Network;
    class FlatteningLayer : public Layer {
    public:
        FlatteningLayer(Vector3<int> _inputSize);
        FlatteningLayer(const JSON &json);
        void CPURun(const Tensor<double> &input) override;
        virtual double getChain(const Vector4<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        virtual std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;
    };
}
#endif //NEURALNETLIBRARY_FLATTENINGLAYER_H
