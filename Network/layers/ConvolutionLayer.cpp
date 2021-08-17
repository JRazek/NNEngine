//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"
#include "../Network.h"

void cn::ConvolutionLayer::run(const Bitmap<float> &bitmap) {
    //convolve bitmap - must have correct sizes etc. Garbage in garbage out.
    std::vector<Bitmap<float>> output;
    output.reserve(4);
    for(int i = 0; i < kernels.size(); i ++){
        Bitmap<float> * kernel = &kernels[i];
        output.push_back(Utils::convolve(*kernel, bitmap, paddingX, paddingY, strideX, strideY));
    }
}

cn::ConvolutionLayer::ConvolutionLayer(int _id, cn::Network *_network, int _kernelSizeX, int _kernelSizeY, int _kernelSizeZ, int _kernelsCount, int _paddingX, int _paddingY,
                                       int _strideX,
                                       int _strideY) :
                                       kernelSizeX(_kernelSizeX),
                                       kernelSizeY(_kernelSizeY),
                                       kernelSizeZ(_kernelSizeZ),
                                       kernelsCount(_kernelsCount),
                                       paddingX(_paddingX),
                                       paddingY(_paddingY),
                                       strideX(_strideX),
                                       strideY(_strideY),
                                       cn::Layer(_id, _network) {
    kernels.reserve(_kernelsCount);
    for(int i = 0; i < _kernelsCount; i ++){
        kernels.emplace_back(_kernelSizeX, _kernelSizeY, _kernelSizeZ);
        std::fill(kernels.back().data(), kernels.back().data() + _kernelSizeX * _kernelSizeY * _kernelSizeZ, 0);
    }
}

