//
// Created by user on 28.08.2021.
//

#ifndef NEURALNETLIBRARY_IDENTITY_H
#define NEURALNETLIBRARY_IDENTITY_H
#include "DifferentiableFunction.h"

struct Identity : public DifferentiableFunction{
    double func(double x) const override;
    double derive(double x) const override;
};


#endif //NEURALNETLIBRARY_IDENTITY_H
