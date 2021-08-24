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
