//
// Created by user on 07.11.2021.
//

#ifndef NEURALFLOWS_SOFTMAX_H
#define NEURALFLOWS_SOFTMAX_H


#include "../Sigmoid/Sigmoid.h"

namespace cn {
    class Softmax : public Sigmoid{
        std::vector<double> dividers;
    public:
        Softmax(Vector3<int> _inputSize);
        Softmax(const JSON &json);
        void CPURun(const Tensor<double> &input) override;
        double getChain(const Vector4<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        std::unique_ptr<Layer> getCopyAsUniquePtr() const noexcept override;

        void resetState() override;
    };

}


#endif //NEURALFLOWS_SOFTMAX_H
