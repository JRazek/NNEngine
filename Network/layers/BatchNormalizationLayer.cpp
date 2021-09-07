//
// Created by user on 20.08.2021.
//

#include "BatchNormalizationLayer.h"
#include "../Network.h"

cn::BatchNormalizationLayer::BatchNormalizationLayer(int _id, Network &_network) : Layer(_id, _network) {
    outputSize = inputSize;
}

cn::Bitmap<double> cn::BatchNormalizationLayer::run(const cn::Bitmap<double> &input) {
    if(input.size() != inputSize)
        throw std::logic_error("invalid bitmap input for normalization layer!");
    Bitmap<double> result(outputSize, input.data());

    double max = 0;

    for(auto it = input.data(); it != input.data() + input.size().multiplyContent(); ++it){
        max = std::max(*it, max);
    }
    if(std::abs(max) > 1) {
        for (auto it = result.data(); it != result.data() + result.size().multiplyContent(); ++it) {
            *it = (*it) / max;
        }
        normalizationFactor = max;
    }else{
        normalizationFactor = 1;
    }
    return result;
}

double cn::BatchNormalizationLayer::getChain(const Vector3<int> &inputPos) {
    if(normalizationFactor == 0)
        return 0;
    return (1.f/normalizationFactor) * network->getChain(__id + 1, inputPos);
}

cn::JSON cn::BatchNormalizationLayer::jsonEncode() const{
    JSON structure;
    structure["id"] = __id;
    structure["type"] = "bnl";
    return structure;
}

cn::BatchNormalizationLayer::BatchNormalizationLayer(cn::Network &_network, const cn::JSON &json): BatchNormalizationLayer(json["id"], _network)
{}
