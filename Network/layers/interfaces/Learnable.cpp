//
// Created by jrazek on 24.08.2021.
//

#include "Learnable.h"

cn::Learnable::Learnable(int id, Vector3<int> _inputSize, int neuronsCount) : Layer(id, _inputSize), neuronsCount(neuronsCount) {}

int cn::Learnable::getNeuronsCount() const{
    return neuronsCount;
}
