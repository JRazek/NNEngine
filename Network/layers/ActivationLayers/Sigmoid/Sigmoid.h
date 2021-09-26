//
// Created by user on 06.09.2021.
//

#ifndef NEURALNETLIBRARY_SIGMOID_H
#define NEURALNETLIBRARY_SIGMOID_H
#include <cmath>
#include "../../interfaces/Layer.h"


namespace cn {
    class Sigmoid : public Layer{
    public:
        constexpr static double e = M_E;
        Sigmoid(int id, Vector3<int> _inputSize);
        Sigmoid(const JSON &json);
        void CPURun(const Tensor<double> &input) override;
        double getChain(const Vector4<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        static double sigmoid(double x);
        static double diff(double x);
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
    };
}

#endif //NEURALNETLIBRARY_SIGMOID_H
