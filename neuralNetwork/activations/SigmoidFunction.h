#pragma once
#include "ActivationFunction.h"
struct SigmoidFunction : ActivationFunction{
    float operator()(float x) const override{
        return 0;
    }
};