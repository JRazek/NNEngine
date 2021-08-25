//
// Created by user on 20.08.2021.
//

#include "MaxPoolingLayer.h"
#include "../Network.h"

cn::MaxPoolingLayer::MaxPoolingLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY) :
        Layer(_id, _network),
        kernelSizeX(_kernelSizeX),
        kernelSizeY(_kernelSizeY){
    int sizeX, sizeY, sizeZ;
    if(__id == 0){
        sizeX = Utils::afterMaxPoolSize(kernelSizeX, network->inputDataWidth);
        sizeY = Utils::afterMaxPoolSize(kernelSizeY, network->inputDataHeight);
        sizeZ = network->inputDataDepth;
    }else{
        const Bitmap<float> *prev = network->getLayers()->at(__id - 1)->getOutput();
        sizeX = Utils::afterMaxPoolSize(kernelSizeX, prev->w());
        sizeY = Utils::afterMaxPoolSize(kernelSizeY, prev->h());
        sizeZ = prev->d();
    }
    output.emplace(Bitmap<float>(sizeX, sizeY, sizeZ));
}

void cn::MaxPoolingLayer::run(const cn::Bitmap<float> &bitmap) {
    Bitmap<float> res = Utils::maxPool(bitmap, kernelSizeX, kernelSizeY);
    if(res.w() != output->w() || res.h() != output->h() || res.d() != output->d()){
        throw std::logic_error("invalid output size in max pool!");
    }
    std::copy(res.data(), res.data() + res.w() * res.h() * res.d(), output->data());
    Layer::run(bitmap);
}

float cn::MaxPoolingLayer::getChain(const Vector3<float> &input) {
    return 0;
}
