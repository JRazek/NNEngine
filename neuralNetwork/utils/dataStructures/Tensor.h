#pragma once
#include "Matrix.h"
struct Tensor{
    std::vector<Matrix> matrices;
    Tensor(int x, int y, int z);
};