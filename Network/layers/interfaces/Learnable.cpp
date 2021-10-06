//
// Created by jrazek on 24.08.2021.
//

#include "Learnable.h"

cn::Learnable::Learnable(Vector3<int> _inputSize, int neuronsCount) : Layer(_inputSize), neuronsCount(neuronsCount) {}

int cn::Learnable::getNeuronsCount() const{
    return neuronsCount;
}

cn::Learnable::Learnable(cn::Vector3<int> _inputSize) : Layer(_inputSize) {}
