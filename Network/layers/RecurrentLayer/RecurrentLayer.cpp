//
// Created by jrazek on 24.09.2021.
//

#include "RecurrentLayer.h"

std::unique_ptr<cn::Layer> cn::RecurrentLayer::getCopyAsUniquePtr() const {
//    return std::unique_ptr<RecurrentLayer>(*this);
}

cn::RecurrentLayer::RecurrentLayer(int _id, cn::Vector3<int> _inputSize) : Layer(_id, _inputSize) {}

void cn::RecurrentLayer::CPURun(const cn::Bitmap<double> &_input) {

}

double cn::RecurrentLayer::getChain(const cn::Vector3<int> &inputPos) {
    return 0;
}
