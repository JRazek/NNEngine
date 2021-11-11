//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_MAXPOOLINGLAYER_H
#define NEURALNETLIBRARY_MAXPOOLINGLAYER_H
#include "../interfaces/Layer.h"

namespace cn {
    class MaxPoolingLayer : public Layer{
    public:
        Vector2<int> kernelSize;

        std::vector<Tensor<Vector2<int>>> mapping;
        MaxPoolingLayer(Vector3<int> _inputSize, cn::Vector2<int> _kernelSize);
        MaxPoolingLayer(const cn::JSON &json);
        void CPURun(const Tensor<double> &input) override;
        double getChain(const Vector4<int> &inputPos) override;
        JSON jsonEncode() const override;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;
        void resetState() override;
        virtual std::unique_ptr<Layer> reproduce(const Layer *net, int seed = 1) const override;
    };
}

#endif //NEURALNETLIBRARY_MAXPOOLINGLAYER_H
