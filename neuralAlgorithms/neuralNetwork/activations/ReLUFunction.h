#pragma once
#include <activations/ActivationFunction.h>
struct ReLUFunction : ActivationFunction{
    ReLUFunction(){};
    float operator()(float x) const override {
        if(x > 0)
            return x;
        return 0;
    }
};