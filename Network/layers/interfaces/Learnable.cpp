//
// Created by jrazek on 24.08.2021.
//

#include "Learnable.h"

cn::Learnable::Learnable(int id, Network &network, int _neuronsCount): Layer(id, network), neuronsCount(_neuronsCount) {}

int cn::Learnable::getNeuronsCount() const{
    return neuronsCount;
}
