#pragma once
#include "Matrix.h"
struct Tensor{
    std::vector<Matrix> matrices;
    int x,y,z;
    Tensor(int x, int y, int z);
    Tensor(int z);
};