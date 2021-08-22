//
// Created by user on 20.08.2021.
//

#include "BatchNormalizationLayer.h"
#include "../Network.h"

cn::BatchNormalizationLayer::BatchNormalizationLayer(int _id, Network &_network) : Layer(_id, _network) {
    int sizeX, sizeY, sizeZ;
    if(id == 0){
        sizeX = network->inputDataWidth;
        sizeY = network->inputDataHeight;
        sizeZ = network->inputDataDepth;
    }else{
        Bitmap<float> *prev = &network->layers[id - 1]->output.value();
        sizeX = prev->w;
        sizeY = prev->h;
        sizeZ = prev->d;
    }
    output.emplace(Bitmap<float>(sizeX, sizeY, sizeZ));
}

void cn::BatchNormalizationLayer::run(const cn::Bitmap<float> &bitmap) {
    if(bitmap.w != output->w || bitmap.h != output->h || bitmap.d != output->d)
        throw std::logic_error("invalid bitmap input for normalization layer!");
    std::copy(bitmap.data(), bitmap.data() + bitmap.w * bitmap.h * bitmap.d, output->data());

    float max = 0;
    auto outputR = &output.value();
    for(auto it = outputR->data(); it != outputR->data() + outputR->w * outputR->h * outputR->d; ++it){
        max = std::max(*it, max);
    }
    for(auto it = outputR->data(); it != outputR->data() + outputR->w * outputR->h * outputR->d; ++it){
        *it = (*it)/max;
    }
    normalizationFactor = max;
}

float cn::BatchNormalizationLayer::getChain(int neuronID) {
    return normalizationFactor;
}
