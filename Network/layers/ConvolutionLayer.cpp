//
// Created by jrazek on 27.07.2021.
//

#include "ConvolutionLayer.h"

void ConvolutionLayer::run(Bitmap *bitmap) {}

ConvolutionLayer::ConvolutionLayer(int id, Network *network, int kernelsCount) : Layer(id, network) {}

Bitmap *ConvolutionLayer::convolve(const Bitmap *kernel, const Bitmap *input, int padding) {
    //convolution here
    return nullptr;
}
