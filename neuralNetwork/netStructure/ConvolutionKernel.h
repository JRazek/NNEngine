#pragma once
#include <vector>

struct ConvolutionKernel{
    struct Matrix{
        int sizeX, sizeY;
        std::vector<float> weights;
    };
    const int id;
    std::vector<Matrix> matrices;
};