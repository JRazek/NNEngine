//
// Created by jrazek on 27.07.2021.
//
#include "ConvolutionLayer.h"
#include "../Network.h"

void cn::ConvolutionLayer::run(const Bitmap<float> &bitmap) {
    //convolve bitmap - must have correct sizes etc. Garbage in garbage out.
    int outW = Utils::afterConvolutionSize(kernelSizeX, bitmap.w, paddingX, strideX);
    int outH = Utils::afterConvolutionSize(kernelSizeY, bitmap.h, paddingY, strideY);
    output = new Bitmap<float>(outW, outH, kernelsCount);

    float *tmp = new float [outW * outH];
    std::fill(tmp, tmp + outW * outH, 0);

    for(int i = 0; i < kernelsCount; i ++){
        //todo FUCKING FIX
        Bitmap<float> layer = Utils::sumBitmapLayers(Utils::convolve(kernels[i], bitmap, paddingX, paddingY, strideX, strideY));
        //std::fill(layer.data(), layer.data() + outW * outH, 0);

        output->setLayer(i, layer.data());
    }
    delete [] tmp;
    for(auto it = output->data(); it != output->data() + outW * outH; ++it){
        *it = activationFunction.func(*it);
    }
}

void cn::ConvolutionLayer::randomInit() {
    for(auto &k : kernels){
        for(auto it = k.data(); it != k.data() + k.w * k.h * k.d; ++it){
            *it = network->getWeightRandom();
        }
    }
}

cn::ConvolutionLayer::ConvolutionLayer(int _id, cn::Network *_network, int _kernelSizeX, int _kernelSizeY,
                                       int _kernelSizeZ,
                                       int _kernelsCount, const DifferentiableFunction &_activationFunction, int _paddingX,
                                       int _paddingY,
                                       int _strideX, int _strideY) :
                                       kernelSizeX(_kernelSizeX),
                                       kernelSizeY(_kernelSizeY),
                                       kernelSizeZ(_kernelSizeZ),
                                       kernelsCount(_kernelsCount),
                                       activationFunction(_activationFunction),
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

