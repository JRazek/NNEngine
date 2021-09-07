//
// Created by user on 06.09.2021.
//

#ifndef NEURALNETLIBRARY_SIGMOID_H
#define NEURALNETLIBRARY_SIGMOID_H
#include <cmath>
#include "../interfaces/Layer.h"


namespace cn {
    class Sigmoid : public Layer{
    public:
        constexpr static double e = M_E;
        Sigmoid(int id, Network &network);
        Sigmoid(Network &_network, const JSON &json);
        Bitmap<double> run(const Bitmap<double> &input) override;
        double getChain(const Vector3<int> &inputPos) override;
        virtual JSON jsonEncode() const override;
        static double sigmoid(double x);
        static double diff(double x);
    };
}

#endif //NEURALNETLIBRARY_SIGMOID_H
