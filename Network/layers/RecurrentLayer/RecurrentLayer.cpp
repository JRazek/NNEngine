//
// Created by jrazek on 24.09.2021.
//

#include "RecurrentLayer.h"

std::unique_ptr<cn::Layer> cn::RecurrentLayer::getCopyAsUniquePtr() const {
    return std::make_unique<RecurrentLayer>(*this);
}

cn::RecurrentLayer::RecurrentLayer(int _id, cn::Vector3<int> _inputSize) : Layer(_id, _inputSize), identity(inputSize) {
    outputSize = inputSize;
    std::fill(identity.data(), identity.data() + identity.size().multiplyContent(), 0);
}

void cn::RecurrentLayer::CPURun(const cn::Tensor<double> &_input) {
    Tensor<double> res = Utils::elementWiseSum(_input, time == 0 ? identity : output[getTime() - 1]);
    output.push_back(res);
}

double cn::RecurrentLayer::getChain(const Vector4<int> &inputPos) {
    //todo testing
    return nextLayer->getChain({inputPos.x, inputPos.y, inputPos.z, getTime() - 1});
}

cn::JSON cn::RecurrentLayer::jsonEncode() const {
    return Layer::jsonEncode();
}

cn::RecurrentLayer::RecurrentLayer(const cn::JSON &json) : RecurrentLayer(json.at("id"), json.at("input_size")) {}

