//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../../Utils/dataStructures/Vector3.h"

cn::Layer::Layer(int _id, Network &_network): __id(_id), network(&_network){}

const cn::Bitmap<float> *cn::Layer::getOutput() {
    return &output.value();
}

int cn::Layer::id() const {
    return __id;
}

void cn::Layer::run(const cn::Bitmap<float> &bitmap) {
    _input = &bitmap;
}
