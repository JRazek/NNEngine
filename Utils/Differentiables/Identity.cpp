//
// Created by user on 28.08.2021.
//

#include "Identity.h"

double Identity::func(double x) const {
    return x;
}

double Identity::derive([[maybe_unused]]double x) const {
    return 1;
}
