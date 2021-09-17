//
// Created by user on 20.08.2021.
//

#include "BatchNormalizationLayer.h"
#include "../../Network.h"

cn::BatchNormalizationLayer::BatchNormalizationLayer(int _id, Vector3<int> _inputSize) : Layer(_id, _inputSize) {
    outputSize = inputSize;
}

void cn::BatchNormalizationLayer::CPURun(const cn::Bitmap<double> &input) {
    if(input.size() != inputSize)
        throw std::logic_error("invalid bitmap input for normalization layer!");
    Bitmap<double> result(outputSize, input.dataConst());

    double max = 0;

    for(auto it = input.dataConst(); it != input.dataConst() + input.size().multiplyContent(); ++it){
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
    output.emplace(std::move(result));
}

double cn::BatchNormalizationLayer::getChain(const Vector3<int> &inputPos) {
    if(normalizationFactor == 0)
        return 0;
    return (1.f/normalizationFactor) * nextLayer->getChain(inputPos);
}

cn::JSON cn::BatchNormalizationLayer::jsonEncode() const{
    JSON structure;
    structure["id"] = __id;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "bnl";
    return structure;
}

cn::BatchNormalizationLayer::BatchNormalizationLayer(const JSON &json) :
BatchNormalizationLayer(json.at("id"), json.at("input_size"))
{}

std::unique_ptr<cn::Layer> cn::BatchNormalizationLayer::getCopyAsUniquePtr() const {
    return std::make_unique<BatchNormalizationLayer>(*this);
}
