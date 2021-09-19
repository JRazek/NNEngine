//
// Created by jrazek on 09.09.2021.
//

#include "InputLayer.h"

cn::InputLayer::InputLayer(int _id, cn::Vector3<int> _inputSize) : Layer(_id, _inputSize) {
    outputSize = inputSize;
}

cn::InputLayer::InputLayer(const cn::JSON &json):Layer(json.at("id"), json.at("input_size"))
{}

void cn::InputLayer::CPURun(const cn::Bitmap<double> &_input) {
    input = std::make_unique<Bitmap<double>>(std::move(_input));
    output = std::make_unique<Bitmap<double>>(std::move(_input));
}

double cn::InputLayer::getChain(const cn::Vector3<int> &inputPos) {
    return nextLayer->getChain(inputPos);
}

cn::JSON cn::InputLayer::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "il";
    return structure;
}

std::unique_ptr<cn::Layer> cn::InputLayer::getCopyAsUniquePtr() const {
    return std::make_unique<InputLayer>(*this);
}

const std::unique_ptr<cn::Bitmap <double>> &cn::InputLayer::getInput() const {
    return input;
}

cn::InputLayer::InputLayer(const cn::InputLayer &inputLayer):Layer(inputLayer) {
     input = std::make_unique<Bitmap<double>>(*inputLayer.getInput().get());
}
