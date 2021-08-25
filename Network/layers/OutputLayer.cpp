//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../Network.h"

cn::OutputLayer::OutputLayer(int id, cn::Network &network) : FlatteningLayer(id, network) {
    int _w = network.getLayers()->at(id -1)->getOutput()->w();
    int _h = network.getLayers()->at(id -1)->getOutput()->h();
    int _d = network.getLayers()->at(id -1)->getOutput()->d();
    output.emplace(_w, _h, _d);
}

void cn::OutputLayer::run(const cn::Bitmap<float> &input) {
    FlatteningLayer::run(input);
}

float cn::OutputLayer::getChain(const Vector3<int> &input) {
    return output->getCell(input) - target->getCell(input);
}

void cn::OutputLayer::setTarget(const cn::Bitmap<float> *_target) {
    target = _target;
}
