//
// Created by jrazek on 24.08.2021.
//

#include "OutputLayer.h"
#include "../Network.h"

float cn::OutputLayer::getChain(int neuronID) {
    if(!target.has_value() || target->w != output->w || target->h != 1 || target->d != 1){
        throw std::logic_error("error in output layer! invalid sizes");
    }
    return output->getCell(neuronID, 0, 0) - target->getCell(neuronID, 0, 0);
}

cn::OutputLayer::OutputLayer(int id, cn::Network &network) : FlatteningLayer(id, network) {

}

float cn::OutputLayer::getError() {
    float sum = 0;
    for(int i = 0; i < target->w; i++){
        sum += 0.5f * pow(output->getCell(i, 0, 0) - target->getCell(i, 0, 0), 2);
    }
    return sum;
}
