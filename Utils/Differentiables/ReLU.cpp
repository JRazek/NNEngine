//
// Created by user on 18.08.2021.
//

#include "ReLU.h"

float ReLU::diff(float x) const {
    if(x > 0)
        return 1;
    return 0;
}

float ReLU::func(float x) const {
    if(x > 0)
        return x;
    return 0;
}
