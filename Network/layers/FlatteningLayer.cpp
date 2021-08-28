//
// Created by jrazek on 19.08.2021.
//

#include "FlatteningLayer.h"
#include "../Network.h"

cn::FlatteningLayer::FlatteningLayer(int _id, Network &_network) : Layer(_id, _network) {
    inputSize = network->getInputSize(_id);
    int size = inputSize.multiplyContent();
    outputSize = Vector3<int>(size, 1, 1);
}

cn::Bitmap<float> cn::FlatteningLayer::run(const cn::Bitmap<float> &input) {
    if(input.size().multiplyContent() != inputSize.multiplyContent())
        throw std::logic_error("invalid input input for flattening layer!");
    return Bitmap<float>({inputSize.multiplyContent(), 1, 1}, input.data());
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
