//
// Created by user on 18.08.2021.
//

#include "Sigmoid.h"
#include <cmath>

double Sigmoid::func(double x) const {
    return 1.f/(1.f + std::pow(std::exp(1), -x));
}

double Sigmoid::derive(double x) const {
    double sig = func(x);
    return sig * (1.f - sig);
}
