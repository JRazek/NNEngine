#pragma once
#include <Layer.h>
#include <ConvolutionKernel.h>
#include <vector>

struct ConvolutionLayer : Layer{
    std::vector<ConvolutionKernel *> kernels;
};