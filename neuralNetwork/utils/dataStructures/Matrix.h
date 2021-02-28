#pragma once

#include <vector>

struct Matrix{
    Matrix(int sizeX, int sizeY);
    int x,y;
    std::vector< std::vector< float > > weights;
};