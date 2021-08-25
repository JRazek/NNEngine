//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../../Utils/dataStructures/Vector3.h"

cn::Layer::Layer(int _id, Network &_network): __id(_id), network(&_network){}

const cn::Bitmap<float> *cn::Layer::getOutput() const {
    return &output.value();
}

int cn::Layer::id() const {
    return __id;
}

