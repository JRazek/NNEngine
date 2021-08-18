//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_SIGMOID_H
#define NEURALNETLIBRARY_SIGMOID_H

#include "DifferentiableFunction.h"

struct Sigmoid : public DifferentiableFunction{
    float func(float x) override;
    float diff(float x) override;
};


#endif //NEURALNETLIBRARY_SIGMOID_H
