//
// Created by jrazek on 19.08.2021.
//

#include "FlatteningLayer.h"
#include "../Network.h"

cn::FlatteningLayer::FlatteningLayer(int _id, Network &_network) : Layer(_id, _network) {
    int size;
    if(__id == 0){
        size = network->inputDataWidth * network->inputDataHeight * network->inputDataDepth;
    }else{
        const Bitmap<float> *prev = network->getLayers()->at(__id - 1)->getOutput();
        size = prev->w() * prev->h() * prev->d();
    }
    output.emplace(Bitmap<float>(size, 1, 1));
}

void cn::FlatteningLayer::run(const cn::Bitmap<float> &bitmap) {
    if(output->w() != bitmap.w() * bitmap.h() * bitmap.d())
        throw std::logic_error("invalid bitmap input for flattening layer!");
    std::copy(bitmap.data(), bitmap.data() + bitmap.w() * bitmap.h() * bitmap.d(), output->data());
    Layer::run(bitmap);
}

float cn::FlatteningLayer::getChain(const Vector3<int> &input) {
    return 0;
}
