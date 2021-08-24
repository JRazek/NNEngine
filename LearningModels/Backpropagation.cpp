//
// Created by user on 21.08.2021.
//

#include "Backpropagation.h"
#include "../Utils/Bitmap.h"
#include "../Network/Network.h"
cn::Backpropagation::Backpropagation(cn::Network &_network): network(_network) {

}

void cn::Backpropagation::propagate(const cn::Bitmap<float> &target) {
    Bitmap<float> &output = network.getOutput();
    if(output.w != target.w || output.h != target.h || output.d != target.d){
        throw std::logic_error("Backpropagation, invalid target!");
    }

}
