//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"
#include "../Network.h"

void cn::ConvolutionLayer::run(const Bitmap<float> &bitmap) {
    //convolve bitmap - must have correct sizes etc. Garbage in garbage out.
    int outW = Utils::afterConvolutionSize(kernelSizeX, bitmap.w, paddingX, strideX);
    int outH = Utils::afterConvolutionSize(kernelSizeY, bitmap.h, paddingY, strideY);
    Layer::output = new Bitmap<float>(outW, outH, kernelsCount);
    for(int i = 0; i < kernels.size(); i ++){
        Bitmap<float> * kernel = &kernels[i];
        output->setLayer(i, Utils::convolve(*kernel, bitmap, paddingX, paddingY, strideX, strideY).data());
    }
}

void cn::ConvolutionLayer::randomInit() {
    for(auto &k : kernels){
        for(auto it = k.data(); it != k.data() + k.w * k.h * k.d; ++it){
            //todo rand float
        }
    }
}

cn::ConvolutionLayer::ConvolutionLayer(int _id, cn::Network *_network, int _kernelSizeX, int _kernelSizeY, int _kernelSizeZ, int _kernelsCount, const std::function<float(float)> &_activation, int _paddingX, int _paddingY,
                                       int _strideX,
                                       int _strideY) :
                                       kernelSizeX(_kernelSizeX),
                                       kernelSizeY(_kernelSizeY),
                                       kernelSizeZ(_kernelSizeZ),
                                       kernelsCount(_kernelsCount),
                                       activation(_activation),
                                       paddingX(_paddingX),
                                       paddingY(_paddingY),
                                       strideX(_strideX),
                                       strideY(_strideY),
                                       cn::Layer(_id, _network) {
    kernels.reserve(_kernelsCount);
    for(int i = 0; i < _kernelsCount; i ++){
        kernels.emplace_back(_kernelSizeX, _kernelSizeY, _kernelSizeZ);
        //std::fill(kernels.back().data(), kernels.back().data() + _kernelSizeX * _kernelSizeY * _kernelSizeZ, 0);
    }
}

