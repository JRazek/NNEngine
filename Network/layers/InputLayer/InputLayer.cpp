//
// Created by jrazek on 09.09.2021.
//

#include "InputLayer.h"

cn::InputLayer::InputLayer(Vector3<int> _inputSize) : Layer(_inputSize) {
    outputSize = inputSize;
}

cn::InputLayer::InputLayer(const cn::JSON &json): Layer(json.at("input_size"))
{}

void cn::InputLayer::CPURun(const cn::Tensor<double> &_input) {
    input.push_back(_input);
    output.push_back(_input);
    addMemoLayer();
}

double cn::InputLayer::getChain(const Vector4<int> &inputPos) {
    return nextLayer->getChain(inputPos);
}

cn::JSON cn::InputLayer::jsonEncode() const {
    JSON structure;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "il";
    return structure;
}

std::unique_ptr<cn::Layer> cn::InputLayer::getCopyAsUniquePtr() const noexcept{
    return std::make_unique<InputLayer>(*this);
}

const cn::Tensor<double> &cn::InputLayer::getInput(int time) const {
    return input[time];
}

void cn::InputLayer::resetState() {
    Layer::resetState();
    input.clear();
}

