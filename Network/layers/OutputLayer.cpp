//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../Network.h"

cn::OutputLayer::OutputLayer(int id, cn::Network &network) : FlatteningLayer(id, network) {}

cn::Bitmap<float> cn::OutputLayer::run(const cn::Bitmap<float> &input) {
    return FlatteningLayer::run(input);
}

float cn::OutputLayer::getChain(const Vector3<int> &input) {
    return network->getOutput(__id)->getCell(input) - target->getCell(input);
}

void cn::OutputLayer::setTarget(const cn::Bitmap<float> *_target) {
    target = _target;
}
