//
// Created by user on 20.08.2021.
//

#include "BatchNormalizationLayer.h"
#include "../Network.h"

cn::BatchNormalizationLayer::BatchNormalizationLayer(int _id, Network &_network) : Layer(_id, _network) {
    outputSize = inputSize;
}

cn::Bitmap<float> cn::BatchNormalizationLayer::run(const cn::Bitmap<float> &input) {
    if(input.size() != inputSize)
        throw std::logic_error("invalid bitmap input for normalization layer!");
    Bitmap<float> result(outputSize, input.data());

    float max = 0;

    for(auto it = input.data(); it != input.data() + input.size().multiplyContent(); ++it){
        max = std::max(*it, max);
    }
    for(auto it = result.data(); it != result.data() + result.size().multiplyContent(); ++it){
        *it = (*it)/max;
    }
    normalizationFactor = max;

    return result;
}

float cn::BatchNormalizationLayer::getChain(const Vector3<int> &inputPos) {
    return (1.f/normalizationFactor) * network->getChain(__id + 1, inputPos);
}
