//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../../Network.h"
#include "CUDAOutputLayer.cuh"

cn::OutputLayer::OutputLayer(int id, Vector3<int> _inputSize) : FlatteningLayer(id, _inputSize) {}

void cn::OutputLayer::CPURun(const cn::Tensor<double> &input) {
    return FlatteningLayer::CPURun(input);
}

double cn::OutputLayer::getChain(const Vector4<int> &input) {
    return getInput(input.t).getCell({input.x, input.y, input.z}) - target->getCell({input.x, input.y, input.z});
}

void cn::OutputLayer::setTarget(const cn::Tensor<double> *_target) {
    if(outputSize != _target->size()){
        throw std::logic_error("");
    }
    target = _target;
}

cn::JSON cn::OutputLayer::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "ol";
    return structure;
}

cn::OutputLayer::OutputLayer(const cn::JSON &json) :
FlatteningLayer(json)
{}

std::unique_ptr<cn::Layer> cn::OutputLayer::getCopyAsUniquePtr() const {
    return std::make_unique<OutputLayer>(*this);
}

void cn::OutputLayer::CUDAAutoGrad() {
    CUDAOutputLayer::CUDAAutoGrad(*this);
}
