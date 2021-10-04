//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../../Network.h"

#ifdef NNL_WITH_CUDA
#include "CUDAOutputLayer.cuh"
void cn::OutputLayer::CUDAAutoGrad() {
    CUDAOutputLayer::CUDAAutoGrad(*this);
}

#endif



cn::OutputLayer::OutputLayer(Vector3<int> _inputSize) : FlatteningLayer(_inputSize) {}

void cn::OutputLayer::CPURun(const cn::Tensor<double> &input) {
    addMemoLayer();
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
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "ol";
    return structure;
}

cn::OutputLayer::OutputLayer(const cn::JSON &json) :
FlatteningLayer(json)
{}

std::unique_ptr<cn::Layer> cn::OutputLayer::getCopyAsUniquePtr() const noexcept{
    return std::make_unique<OutputLayer>(*this);
}
