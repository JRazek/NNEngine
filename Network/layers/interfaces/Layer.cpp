//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../Network.h"

cn::Layer::Layer(int _id, Network &_network): __id(_id), network(&_network){
    inputSize = network->getInputSize(_id);
    memoizationStates.emplace(Bitmap<bool>(inputSize));
    memoizationTable.emplace(Bitmap<double>(inputSize));
    resetMemoization();
}

[[maybe_unused]] int cn::Layer::id() const {
    return __id;
}

void cn::Layer::setMemo(const Vector3<int> &pos, double val) {
    memoizationStates->setCell(pos, true);
    memoizationTable->setCell(pos, val);
}

void cn::Layer::resetMemoization() {
    std::fill(memoizationStates->data(), memoizationStates->data() + memoizationStates->size().multiplyContent(), false);
}

double cn::Layer::getMemo(const Vector3<int> &pos) const {
    return memoizationTable->getCell(pos);
}

bool cn::Layer::getMemoState(const Vector3<int> &pos) const {
    return memoizationStates->getCell(pos);
}

Vector3<int> cn::Layer::getOutputSize() const {
    return outputSize;
}

