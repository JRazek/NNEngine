//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_SIGMOID_H
#define NEURALNETLIBRARY_SIGMOID_H

#include "DifferentiableFunction.h"

struct Sigmoid : public DifferentiableFunction{
    float func(float x) const override;
    float diff(float x) const override;
};


#endif //NEURALNETLIBRARY_SIGMOID_H
