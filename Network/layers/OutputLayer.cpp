//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"


cn::OutputLayer::OutputLayer(int id, cn::Network &network) : FlatteningLayer(id, network) {}

void cn::OutputLayer::run(const cn::Bitmap<float> &input) {
    Layer::run(input);
}

float cn::OutputLayer::getChain(const Vector3<int> &input) {
    return 0;
}
