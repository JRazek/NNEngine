#pragma once
#include <activations/ActivationFunction.h>
#include <cmath>
struct SigmoidFunction : ActivationFunction{
    SigmoidFunction(){};
    constexpr static float e = 2.718281828f;
    
    float operator()(float x) const override{
        return 1.f/(1 + pow(e, -x));
    }
};