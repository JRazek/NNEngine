//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_SIGMOID_H
#define NEURALNETLIBRARY_SIGMOID_H

#include "DifferentiableFunction.h"
#include <cmath>
struct Sigmoid : public DifferentiableFunction{
    const double e = std::exp(1.0);
    double func(double x) const override;
    double derive(double x) const override;
};


#endif //NEURALNETLIBRARY_SIGMOID_H
