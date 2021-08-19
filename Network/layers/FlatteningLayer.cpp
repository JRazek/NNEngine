//
// Created by jrazek on 19.08.2021.
//

#include "FlatteningLayer.h"

cn::FlatteningLayer::FlatteningLayer(int _id, cn::Network *_network) : Layer(_id, _network) {}

void cn::FlatteningLayer::run(const cn::Bitmap<float> &bitmap) {
    output = new Bitmap<float>(bitmap.w * bitmap.h * bitmap.d, 1, 1, bitmap.data());
}
