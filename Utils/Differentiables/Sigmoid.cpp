//
// Created by user on 18.08.2021.
//

#include "Sigmoid.h"
#include <cmath>

float Sigmoid::func(float x) const {
    return 1.f/(1.f + std::pow(std::exp(1), -x));
}

float Sigmoid::derive(float x) const {
    float sig = func(x);
    return sig * (1.f - sig);
}
