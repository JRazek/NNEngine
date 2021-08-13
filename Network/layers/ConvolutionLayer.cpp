//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"
#include "../Network.h"

void cn::ConvolutionLayer::run(const Bitmap<float> &bitmap) {
    int sizeX = this->network->inputDataWidth;
    int sizeY = this->network->inputDataHeight;

    Bitmap<float> result = cn::Utils::resize(bitmap, sizeX, sizeY);

    //todo!
    //https://stackoverflow.com/questions/6133957/image-downsampling-algorithms

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

