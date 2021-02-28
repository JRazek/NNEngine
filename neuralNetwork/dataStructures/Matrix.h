#pragma once

#include <vector>

struct Matrix{
    Matrix(int sizeX, int sizeY);
    std::vector< std::vector< float > > weights;
};