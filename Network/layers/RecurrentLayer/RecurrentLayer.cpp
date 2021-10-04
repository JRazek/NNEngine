//
// Created by jrazek on 24.09.2021.
//

#include "RecurrentLayer.h"

std::unique_ptr<cn::Layer> cn::RecurrentLayer::getCopyAsUniquePtr() const noexcept{
    return std::make_unique<RecurrentLayer>(*this);
}

cn::RecurrentLayer::RecurrentLayer(cn::Vector3<int> _inputSize) : Layer(_inputSize), identity(inputSize) {
    outputSize = inputSize;
    std::fill(identity.data(), identity.data() + identity.size().multiplyContent(), 0);
}

void cn::RecurrentLayer::CPURun(const cn::Tensor<double> &_input) {
    Tensor<double> res = Utils::elementWiseSum(_input, _time == 0 ? identity : output[getTime() - 1]);
    output.push_back(res);
}

double cn::RecurrentLayer::getChain(const Vector4<int> &inputPos) {
    //todo testing.
    return nextLayer->getChain({inputPos.x, inputPos.y, inputPos.z, getTime() - 1});
}

cn::JSON cn::RecurrentLayer::jsonEncode() const {
    JSON structure;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "rcl";
    structure["internal_layers"] = std::vector<JSON>();
    for(auto &l : internalLayers){
        structure["internal_layers"].push_back(l->jsonEncode());
    }
    return structure;
}

cn::RecurrentLayer::RecurrentLayer(const cn::JSON &json) :
RecurrentLayer(Vector3<int>(json.at("input_size"))) {
    for(auto &layerJSON : json.at("internal_layers")){
        std::unique_ptr<Layer> layer = Layer::fromJSON(layerJSON);
        internalLayers.push_back(std::move(layer));
    }
}

cn::RecurrentLayer::RecurrentLayer(const cn::RecurrentLayer &recurrentLayer): Layer(recurrentLayer) {
    for(const std::unique_ptr<Layer> &l : recurrentLayer.internalLayers){
        internalLayers.push_back(l.get()->getCopyAsUniquePtr());
    }
}

