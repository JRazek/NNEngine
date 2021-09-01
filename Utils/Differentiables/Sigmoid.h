//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_SIGMOID_H
#define NEURALNETLIBRARY_SIGMOID_H

#include "DifferentiableFunction.h"

struct Sigmoid : public DifferentiableFunction{
    double func(double x) const override;
    double derive(double x) const override;
};


#endif //NEURALNETLIBRARY_SIGMOID_H
