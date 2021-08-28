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
        const Bitmap<float> &prev = network->getInput(__id);
        size = prev.w() * prev.h() * prev.d();
    }
    output.emplace(Bitmap<float>(size, 1, 1));
}

void cn::FlatteningLayer::run(const cn::Bitmap<float> &input) {
    _input = &input;
    if(output->w() != input.w() * input.h() * input.d())
        throw std::logic_error("invalid input input for flattening layer!");
    std::copy(input.data(), input.data() + input.w() * input.h() * input.d(), output->data());
}

float cn::FlatteningLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    int outputIndex = _input->getDataIndex(inputPos);
    float res = network->getChain(__id + 1, {outputIndex, 0, 0});
    setMemo(inputPos, res);
    return res;
}
