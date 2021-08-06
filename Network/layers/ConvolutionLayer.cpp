//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"

void cn::ConvolutionLayer::run(cn::Bitmap<float> &bitmap) {

}

cn::ConvolutionLayer::ConvolutionLayer(int id, cn::Network *network, int kernelSizeX, int kernelSizeY, int kernelSizeZ, int kernelsCount, int paddingX, int paddingY,
                                       int strideX,
                                       int strideY) :
                                       kernelSizeX(kernelSizeX),
                                       kernelSizeY(kernelSizeY),
                                       kernelSizeZ(kernelSizeZ),
                                       kernelsCount(kernelsCount),
                                       paddingX(paddingX),
                                       paddingY(paddingY),
                                       strideX(strideX),
                                       strideY(strideY),
                                       cn::Layer(id, network) {
    kernels.reserve(kernelsCount);
    for(int i = 0; i < kernelsCount; i ++){
        kernels.emplace_back(kernelSizeX, kernelSizeY, kernelSizeZ);
        std::fill(kernels.back().data(), kernels.back().data() + kernelSizeX * kernelSizeY * kernelSizeZ, 0);
    }
}

cn::Bitmap<float> cn::ConvolutionLayer::convolve(const Bitmap<float> &kernel, const Bitmap<float> &input,
                                                 int paddingX, int paddingY, int stepX, int stepY) {
    if(!(kernel.w % 2 && kernel.h % 2 && kernel.d == input.d)){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = ConvolutionLayer::afterConvolutionSize(kernel.w, input.w, paddingX, stepX);
    int sizeY = ConvolutionLayer::afterConvolutionSize(kernel.h, input.h, paddingY, stepY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }

    cn::Bitmap<float> bitmap (sizeX, sizeY, 1);
    //convolution here
    return bitmap;
}

int cn::ConvolutionLayer::afterConvolutionSize(int kernelSize, int inputSize, int padding, int step) {
    return (inputSize + 2*padding - kernelSize) / step + 1;
}

