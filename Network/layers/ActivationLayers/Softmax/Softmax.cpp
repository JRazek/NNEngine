//
// Created by user on 07.11.2021.
//

#include "Softmax.h"

cn::Softmax::Softmax(cn::Vector3<int> _inputSize) : Sigmoid(_inputSize)
{}

cn::Softmax::Softmax(const cn::JSON &json) : Sigmoid(json)
{}

void cn::Softmax::CPURun(const cn::Tensor<double> &input) {
    Sigmoid::CPURun(input);
    double sum = 0;
    Tensor<double> &output = this->output.back();
    for(auto it = output.dataConst(); it != output.dataConst()+output.size().multiplyContent(); ++it){
        sum += *it;
    }
    output /= sum;
    dividers.push_back(sum);
}

double cn::Softmax::getChain(const cn::Vector4<int> &inputPos) {
    return Sigmoid::getChain(inputPos) * dividers[inputPos.t];
}

cn::JSON cn::Softmax::jsonEncode() const {
    return Sigmoid::jsonEncode();
}

std::unique_ptr<cn::Layer> cn::Softmax::getCopyAsUniquePtr() const noexcept {
    return Sigmoid::getCopyAsUniquePtr();
}

void cn::Softmax::resetState() {
    Layer::resetState();
    dividers.clear();
}
