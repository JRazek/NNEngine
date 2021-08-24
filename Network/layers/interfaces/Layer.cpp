//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"

cn::Layer::Layer(int _id, Network &_network): id(_id), network(&_network){}

const cn::Bitmap<float> *cn::Layer::getOutput() {
    return &output.value();
}
