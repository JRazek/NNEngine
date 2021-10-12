//
// Created by j.razek on 07.10.2021.
//

#include "RecurrentOutputLayer.h"

    cn::RecurrentOutputLayer::RecurrentOutputLayer(const Vector3<int> &_inputSize, RecurrentLayer &_parentLayer):
Layer(_inputSize), parentLayer(&_parentLayer) {
    outputSize = inputSize;
}

void cn::RecurrentOutputLayer::CPURun(const cn::Tensor<double> &_input) {
    output.push_back(_input);
}


cn::JSON cn::RecurrentOutputLayer::jsonEncode() const {
    JSON structure;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "rcol";
    return structure;
}

double cn::RecurrentOutputLayer::getChain(const cn::Vector4<int> &inputPos) {
    return parentLayer->getChainFromChild(inputPos);
}

std::unique_ptr<cn::Layer> cn::RecurrentOutputLayer::getCopyAsUniquePtr() const noexcept {
    //todo fix - copies pointer to parent!!!
    return std::make_unique<RecurrentOutputLayer>(*this);
}
