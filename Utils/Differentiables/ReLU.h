//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_RELU_H
#define NEURALNETLIBRARY_RELU_H

#include "DifferentiableFunction.h"

struct ReLU : public DifferentiableFunction{
    float func(float x) const override;
    float diff(float x) const override;
};


#endif //NEURALNETLIBRARY_RELU_H
