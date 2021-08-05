//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"

void cn::ConvolutionLayer::run(cn::Bitmap<float> *bitmap) {}

cn::ConvolutionLayer::ConvolutionLayer(int id, cn::Network *network, int kernelsCount) : cn::Layer(id, network) {}

cn::Bitmap<float> *
cn::ConvolutionLayer::convolve(const cn::Bitmap<float> *kernel, const cn::Bitmap<float> *input, int paddingX = 0, int paddingY = 0, int stepX = 1,
                               int stepY = 1) {
    if(!(kernel->w % 2 && kernel->h % 2 && kernel->d == input->d)){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = ConvolutionLayer::afterConvolutionSize(kernel->w, input->w, paddingX, stepX);
    int sizeY = ConvolutionLayer::afterConvolutionSize(kernel->h, input->h, paddingY, stepY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }

    cn::Bitmap<float> *bitmap = new cn::Bitmap<float>(sizeX, sizeY, 1);
    //convolution here
    return bitmap;
}

int cn::ConvolutionLayer::afterConvolutionSize(int kernelSize, int inputSize, int padding, int step) {
    return (inputSize + 2*padding - kernelSize) / step + 1;
}

