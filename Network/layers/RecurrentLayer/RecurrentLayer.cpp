//
// Created by jrazek on 24.09.2021.
//

#include "RecurrentLayer.h"

std::unique_ptr<cn::Layer> cn::RecurrentLayer::getCopyAsUniquePtr() const {
    RecurrentLayer recurrentLayer(*this);
//    return std::make_unique<RecurrentLayer>(recurrentLayer);
}

cn::RecurrentLayer::RecurrentLayer(int _id, cn::Vector3<int> _inputSize) : Layer(_id, _inputSize) {
    outputSize = inputSize;
    Bitmap<double> identity(inputSize);
    std::fill(identity.data(), identity.data() + identity.size().multiplyContent(), 0);
    memoryStates.push(std::move(identity));
}

void cn::RecurrentLayer::CPURun(const cn::Bitmap<double> &_input) {
    Bitmap<double> res = Utils::elementWiseProduct(_input, memoryStates.top());
    memoryStates.push(res);
    output = std::make_unique<Bitmap<double>>(std::move(res));
}

double cn::RecurrentLayer::getChain(const cn::Vector3<int> &inputPos) {
    return 0;
}

cn::JSON cn::RecurrentLayer::jsonEncode() const {
    return Layer::jsonEncode();
}

void cn::RecurrentLayer::CUDAAutoGrad() {
    Layer::CUDAAutoGrad();
}
