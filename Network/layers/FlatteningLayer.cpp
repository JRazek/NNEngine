//
// Created by jrazek on 19.08.2021.
//

#include "FlatteningLayer.h"
#include "../Network.h"

cn::FlatteningLayer::FlatteningLayer(int _id, cn::Network *_network) : Layer(_id, _network) {
    int size;
    if(id == 0){
        size = network->inputDataWidth * network->inputDataHeight * network->inputDataDepth;
    }else{
        auto prev = network->layers[id - 1]->output;
        size = prev->w * prev->h * prev->d;
    }
    output.emplace(Bitmap<float>(size, 1, 1));
}

void cn::FlatteningLayer::run(const cn::Bitmap<float> &bitmap) {
    std::copy(bitmap.data(), bitmap.data() + bitmap.w * bitmap.h * bitmap.d, output->data());
}
