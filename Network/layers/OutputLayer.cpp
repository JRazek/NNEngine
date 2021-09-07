//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../Network.h"

cn::OutputLayer::OutputLayer(int id, cn::Network &network) : FlatteningLayer(id, network) {}

cn::Bitmap<double> cn::OutputLayer::run(const cn::Bitmap<double> &input) {
    return FlatteningLayer::run(input);
}

double cn::OutputLayer::getChain(const Vector3<int> &input) {
    return network->getOutput(__id).getCell(input) - target->getCell(input);
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
    structure["type"] = "ol";
    return structure;
}

cn::OutputLayer::OutputLayer(cn::Network &_network, const cn::JSON &json) : FlatteningLayer(_network, json)
{}
