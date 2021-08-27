//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../../Utils/dataStructures/Vector3.h"
#include "../../Network.h"

cn::Layer::Layer(int _id, Network &_network): __id(_id), network(&_network){
    const Bitmap<float> *prev = __id == 0 ? network->getInput() : network->getLayers()->at(__id - 1)->getOutput();
    memoizationStates.emplace(prev->w(), prev->h(), prev->d());
    memoizationTable.emplace(prev->w(), prev->h(), prev->d());
}

const cn::Bitmap<float> *cn::Layer::getOutput() const {
    return &output.value();
}

int cn::Layer::id() const {
    return __id;
}

void cn::Layer::setMemo(const Vector3<int> &pos, float val) {
    memoizationStates->setCell(pos, true);
    memoizationTable->setCell(pos, val);
}

void cn::Layer::resetMemoization() {
    std::fill(memoizationStates->data(), memoizationStates->data() + memoizationStates->w() * memoizationStates->h() * memoizationStates->d(), false);}

float cn::Layer::getMemo(const Vector3<int> &pos) const {
    return memoizationTable->getCell(pos);
}

bool cn::Layer::getMemoState(const Vector3<int> &pos) const {
    return memoizationStates->getCell(pos);
}

