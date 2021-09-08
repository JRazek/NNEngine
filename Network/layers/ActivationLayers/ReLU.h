//
// Created by user on 06.09.2021.
//

#ifndef NEURALNETLIBRARY_RELU_H
#define NEURALNETLIBRARY_RELU_H
#include "../interfaces/Layer.h"


namespace cn {
    class ReLU : public Layer{
    public:
        ReLU(int id, Vector3<int> _inputSize);
        ReLU(const JSON &json);
        Bitmap<double> run(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        static double relu(double x);
        static double diff(double x);
        std::unique_ptr<Layer> getCopyAsUniquePtr() const override;
    };
}

#endif //NEURALNETLIBRARY_RELU_H
