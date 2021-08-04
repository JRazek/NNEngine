//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"

void ConvolutionLayer::run(Bitmap *bitmap) {}

ConvolutionLayer::ConvolutionLayer(int id, Network *network, int kernelsCount) : Layer(id, network) {}

Bitmap *ConvolutionLayer::convolve(const Bitmap *kernel, const Bitmap *input, int paddingX = 0, int paddingY = 0, int stepX = 1,
                                   int stepY = 1) {
    if(!(kernel->w % 2 && kernel->h % 2 && kernel->d == input->d)){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = ConvolutionLayer::afterConvolutionSize(kernel->w, input->w, paddingX, stepX);
    int sizeY = ConvolutionLayer::afterConvolutionSize(kernel->h, input->h, paddingY, stepY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }

    Bitmap *bitmap = new Bitmap(sizeX, sizeY, 1);
    //convolution here
    return bitmap;
}

int ConvolutionLayer::afterConvolutionSize(int kernelSize, int inputSize, int padding, int step) {
    return (inputSize + 2*padding - kernelSize) / step + 1;
}

