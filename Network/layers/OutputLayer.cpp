//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../Network.h"

cn::OutputLayer::OutputLayer(int id, Vector3<int> _inputSize) : FlatteningLayer(id, _inputSize) {}

cn::Bitmap<double> cn::OutputLayer::run(const cn::Bitmap<double> &input) {
    return FlatteningLayer::run(input);
}

double cn::OutputLayer::getChain(const Vector3<int> &input) {
    return getInput().value().getCell(input) - target->getCell(input);
}

void cn::OutputLayer::setTarget(const cn::Bitmap<double> *_target) {
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
