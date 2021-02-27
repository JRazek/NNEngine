#pragma once
#include "ActivationFunction.h"
struct ReLUFunction : ActivationFunction{
    float operator()(float x) const override {
        if(x > 0)
            return x;
        return 0;
    }
};